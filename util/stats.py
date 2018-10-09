from collections import OrderedDict
import os
import torch

from . import checkpoint


def data_distribution_hooks(model, activations=True, weights=True,
                            gradients=True):
    """Register hooks to print activations, weights and gradients. Input is
    the root model."""
    # Only interested in bottom-level modules. Otherwise we'll print the same
    # inputs/outputs multiple times.
    leaves = (module for module in model.modules()
              if len(list(module.children())) == 0)
    for module in leaves:
        if activations:
            module.register_forward_hook(_print_activations)
        if weights:
            module.register_forward_hook(_print_weights)
        if gradients:
            module.register_backward_hook(_print_gradients)


def _print_activations(module, activation_input, activation_output):
    """A forward hook to be called whenever a module finishes computing an
    output. Apply to a module using 
        
        module.register_forward_hook(_print_activations)
    
    Prints the size, mean and standard deviation of the output activations."""
    print("Activations:", _get_stats(activation_output))


def _print_weights(module, activation_input, activation_output):
    """A forward hook to be called whenever a module finishes computing an
    output. Apply to a module using 
        
        module.register_forward_hook(_print_weights)
    
    Prints the size, mean and standard deviation of the layer's weights."""
    for params in module.parameters():
        print("Weights:", _get_stats(params))


def _print_gradients(module, grad_input, grad_output):
    """A backward hook to be called whenever a module finishes computing its
    gradients. Apply to a module using 
        
        module.register_backward_hook(_print_gradients)
    
    Prints the size, mean and standard deviation of the output gradients."""
    for tensor in grad_input:
        if tensor is not None:
            print("Gradients:", _get_stats(tensor))


def _get_stats(tensor):
    return str.format("{0}\tmean: {1:.6f}\tstd: {2:.6f}\tmax: {3:.6f}",
                      tensor.size(), tensor.mean().data[0], 
                      tensor.std().data[0], tensor.max().data[0]) 


def data_dump_hooks(model, directory, activations=True, weights=True, 
                    gradients=True):
    """Register hooks to dump tensors to files."""
    # Only interested in bottom-level modules. Otherwise we'll print the same
    # inputs/outputs multiple times.
    leaves = (module for module in model.modules()
              if len(list(module.children())) == 0)
    for i, module in enumerate(leaves):
        name = format(i, "03") + "_" + type(module).__name__
    
        if activations:
            module.register_forward_hook(_dump_activations(directory, name))
        if gradients:
            module.register_backward_hook(_dump_gradients(directory, name))

    if weights:
        _dump_weights(directory, model)


def _dump_activations(directory, layer_name):
    directory = os.path.join(directory, "activations")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    def inner_fn(module, activation_input, activation_output):
        checkpoint.save_tensor(directory, layer_name, activation_output)
    
    return inner_fn


def _dump_gradients(directory, layer_name):
    directory = os.path.join(directory, "gradients")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    def inner_fn(module, grad_input, grad_output):
        for i, tensor in enumerate(grad_input):
            if tensor is None:
                continue
                
            name = layer_name + "_" + str(i)
            checkpoint.save_tensor(directory, name, tensor)
    
    return inner_fn


def _dump_weights(directory, model):
    directory = os.path.join(directory, "weights")
    if not os.path.exists(directory):
        os.makedirs(directory)

    leaves = [module for module in model.modules()
              if len(list(module.children())) == 0]
    for i, module in enumerate(leaves):
        layer_name = format(i, "03") + "_" + type(module).__name__

        for key, value in module.state_dict().items():
            name = layer_name + "_" + key
            checkpoint.save_tensor(directory, name, value)


def data_restore(directory, model):
    """Reverse the effect of `_dump_weights` by loading individual tensors
    into the model."""
    print("Replacing model state with data from", directory)

    leaves = [module for module in model.modules()
              if len(list(module.children())) == 0]
    for i, module in enumerate(leaves):
        layer_name = format(i, "03") + "_" + type(module).__name__

        for key, value in module.state_dict().items():
            name = layer_name + "_" + key
            checkpoint.load_tensor(directory, name, value)


def gradient_distribution_hooks(model):
    """Like the above `_print_gradients`, but aggregates information over a
    whole epoch, and gives more detail about the distribution."""
    
    # Accumulator maps modules to their gradients.
    global gradient_accumulator
    gradient_accumulator = OrderedDict()
    
    # Only supports convolution layers for now.
    conv = (m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    for module in conv:
        module.register_backward_hook(_collect_gradients)
        gradient_accumulator[module] = []


def _collect_gradients(module, grad_input, grad_output):
    """Get the gradients, and append them to a tensor for this module only."""
    
    # I'm still not totally sure whether grad_output is the gradients of the
    # module's outputs, or the gradients being outputted by this module.
    # I assume the former - I want the gradients of the output activations.
    global gradient_accumulator
    assert len(grad_output) == 1
    for gradients in grad_output:
        assert module in gradient_accumulator
        # Ideally here we would save all data and compute statistics once we
        # have all of it. That takes too long and uses too much memory, so I
        # cheat and compute statistics for each batch, then combine the
        # statistics for each batch at the end. This is not mathematically
        # correct, but assuming each batch comes from the same distribution, it
        # should give a reasonable approximation.
        percentiles = _get_percentiles(gradients.data)
        gradient_accumulator[module].append(percentiles)
#        gradient_accumulator[module].append(gradients.data)


def _get_percentiles(tensor):
    """Return the value at every 10th percentile from a given dataset."""
    # Flatten the tensor and sort it.
    tensor = tensor.view(-1)
    tensor, positions = torch.sort(tensor.abs())
    
    # Access every 10th percentile.
    step = max(len(tensor)//10, 1)
    percentiles = list(tensor[::step])
    if len(percentiles) < 11:
        percentiles.append(tensor[-1])
    
    assert len(percentiles) == 11
    return percentiles


def get_gradient_stats():
    """Get a summary of the accumulated gradients.
    
    Returns a list of strings, one per line. Each line contains the gradient
    at every 10th percentile, ordered by absolute size."""
    
    text = []
    
    global gradient_accumulator
    for module, stats in gradient_accumulator.items():
        # Stats should be a 2D array, where each subarray contains the 0th,
        # 10th, 20th, ... percentiles for each batch.
        percentiles = []
        
        # Transpose stats so each subarray contains similar information. All
        # 0th percentile, all 10th percentile, etc.
        stats = zip(*stats)
        
        # Get the 0th percentile of the 0th percentile array, the 10th
        # percentile of the 10th percentile array and so on. This approximates
        # the percentiles of the whole dataset, given only percentiles of
        # smaller subsets.
        for index, array in enumerate(stats):
            percentile = 10 * index
            position = (len(array) * percentile) // 100
            
            array = sorted(array)
            
            if position >= len(array):
                result = str(array[-1])
            else:
                result = str(array[position])
            
            percentiles.append(result)
            
        text.append(" ".join(percentiles))
            
    # Clear the gradient accumulator so it can be used again (but keep the
    # keys/modules in the same order).
    for module in gradient_accumulator:
        gradient_accumulator[module] = []
    
    return text


num_weights = 0
num_operations = 0


def computation_cost_hooks(model, count_weights=True, count_operations=True):
    """Collect data on the number of weights and computations used in a single
    forward pass of the given network.
    
    Looks exclusively at convolution layers (and fully-connected layers which
    have been replaced by convolutions).
     * Batch norm is excluded as it can be merged with other layers at zero cost
     * Pooling is excluded for having negligible cost
     * Activation functions are excluded for having negligible cost
     * Data movement is excluded as it can potentially be avoided"""
    
    modules = (m for m in model.modules() if hasattr(m, "num_operations"))
    
    global num_weights, num_operations
    num_weights = 0
    num_operations = 0
    
    # TODO: can get some double-counting here if there are multiple GPUs.
    print("Warning: figures are only accurate if a single GPU is used.")

    for module in modules:
        if count_weights:
            num_weights += module.num_weights()
        if count_operations:
            module.register_forward_hook(_count_operations)


def _count_operations(module, in_data, out_data):
    """Ask the module how much computation it needed to do, given the shapes of
    its input and output."""
    global num_operations
    assert len(in_data) == 1
    num_operations += module.num_operations(in_data[0].size(), out_data.size())


def computation_costs():
    """Return the values computed here."""
    global num_operations, num_weights
    return num_operations, num_weights
