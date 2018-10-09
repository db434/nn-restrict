"""Module with various options for controlling number formats in the network.

TODO:
 * Weights/activations
"""
 
import torch
import torch.nn.functional


def restrict_gradients(model, minimum=0.0, maximum=0.0, noise=0.0,
                       precision=0.0):
    """Place restrictions on possible gradient values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        fn = _wrap_for_gradients(modifiers)
        
        # Only interested in bottom-level modules. Otherwise we might apply the
        # same transformation multiple times.
        leaves = (module for module in model.modules()
                  if len(list(module.children())) == 0)
                  
        for module in leaves:
            module.register_backward_hook(fn)


def restrict_activations(model, minimum=0.0, maximum=0.0, noise=0.0,
                         precision=0.0):
    """Place restrictions on possible activation values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        fn = _wrap_for_activations(modifiers)
        
        # Only interested in bottom-level modules. Otherwise we might apply the
        # same transformation multiple times.
        leaves = (module for module in model.modules()
                  if len(list(module.children())) == 0)
                  
        for module in leaves:
            module.register_forward_hook(fn)


def restrict_weights(model, minimum=0.0, maximum=0.0, noise=0.0, precision=0.0):
    """Place restrictions on possible weight values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        fn = _wrap_for_weights(modifiers)
        
        # Only interested in bottom-level modules. Otherwise we might apply the
        # same transformation multiple times.
        leaves = (module for module in model.modules()
                  if len(list(module.children())) == 0)
                  
        for module in leaves:
            module.register_forward_hook(fn)


def _get_modifiers(minimum, maximum, noise, precision):
    """Return a list of functions which each receive a Tensor and return a
    Tensor of the same size."""
    
    # Functions will be applied in the order that they're added to this list.
    # Noise should definitely be added before any rounding, but I'm not sure
    # whether precision or thresholding should come first. I think precision.
    modifiers = []
    
    if noise > 0.0:
        modifiers.append(noise_fn(noise))
    
    if precision > 0.0:
        modifiers.append(precision_fn(precision))
    
    if minimum > 0.0:
        modifiers.append(stochastic_threshold_fn(minimum))
    
    if maximum > 0.0:
        modifiers.append(cap_fn(maximum))
        
    return modifiers


def noise_fn(magnitude):
    """Returns a function which takes a Tensor as input and returns a new Tensor
    with random noise added. The amount of noise is in the range +/- magnitude.
    """
    def add_noise(tensor):
        if tensor.is_cuda:
            noise = torch.cuda.FloatTensor(tensor.size())
        else:
            noise = torch.FloatTensor(tensor.size())
        noise.uniform_(-magnitude, magnitude)
        noise = torch.autograd.Variable(noise, requires_grad=False)
        
        return tensor + noise
    
    return add_noise


def precision_fn(precision):
    """Returns a function which takes a Tensor as input and returns a new Tensor
    where all elements are the nearest multiple of `precision`."""
    def round_to_precision(tensor):
        multiples = tensor / precision
        multiples = torch.round(multiples)
        return multiples * precision
    
    return round_to_precision


def threshold_fn(value):
    """Returns a function which takes a Tensor as input and returns a new Tensor
    in which all values with magnitudes less than `value` have been replaced
    with 0."""
    # I don't think there's a way to apply a threshold from below, so I cheat.
    def apply_threshold(tensor):
        signs = torch.sign(tensor)
        magnitudes = torch.abs(tensor)
        magnitudes = torch.nn.functional.threshold(magnitudes, value, 0.0)
        return magnitudes * signs
    
    return apply_threshold


def stochastic_threshold_fn(value):
    """Returns a function which takes a Tensor as input and returns a new Tensor
    in which all values with magnitudes less than `value` have a chance of being
    replaced with 0. The probability for an input of i is `(value-i)/value`. All
    remaining elements less than `value` are replaced by `value`."""
    # I don't think there's a way to apply a threshold from below, so I cheat.
    def apply_threshold(tensor):
        # Decompose inputs into signs and magnitudes to make thresholding
        # easier.
        signs = torch.sign(tensor)
        magnitudes = torch.abs(tensor)
        
        # Create a random number for each input. Remove an input if it is less
        # than its random value. Random values have a maximum of `value`, so all
        # inputs larger than the threshold will remain.
        if tensor.is_cuda:
            rand = torch.cuda.FloatTensor(magnitudes.size())
        else:
            rand = torch.FloatTensor(magnitudes.size())
        rand.uniform_(0.0, value)
        rand = torch.autograd.Variable(rand, requires_grad=False)
        mask_low = magnitudes.lt(rand)
        
        # Set an input to `value` if it is less than the threshold, but not
        # to be set to zero. I actually set all values less than `value` to
        # `value`, and then set a subset of those to zero.
        mask_high = magnitudes.lt(value)
        
        magnitudes.masked_fill_(mask_high, value)
        magnitudes.masked_fill_(mask_low, 0.0)
        
        return magnitudes * signs
    
    return apply_threshold


def cap_fn(value):
    """Returns a function which takes a Tensor as input and returns a new Tensor
    where any elements with magnitude larger than `value` have been replaced
    with +/-`value`."""
    def apply_cap(tensor):
        return torch.clamp(tensor, -value, value)
    
    return apply_cap


def _wrap_for_gradients(functions):
    """Wraps a list of Tensor transformation functions to create a single
    function call which provides all necessary arguments for a backward hook.
    Can then be applied using `register_backward_hook(fn)`.
    """
    assert len(functions) > 0
    
    def backward_hook(module, grad_in, grad_out):
        return _map_tensor(functions, grad_in)
    
    return backward_hook


def _wrap_for_activations(functions):
    """Wraps a list of Tensor transformation functions to create a single
    function call which provides all necessary arguments for a forward hook.
    Can then be applied using `register_forward_hook(fn)`.
    """
    print("Activation modification not currently supported.")
    exit(1)

    # TODO
    # So, while backward_hooks allow modified gradients to be returned,
    # forward_hooks do not allow activations to be modified.
    # Will probably need to add a new Module which does this sort of thing.
    
    assert len(functions) > 0
    
    def forward_hook(module, act_in, act_out):
        return _map_tensor(functions, act_out)
    
    return forward_hook


def _wrap_for_weights(functions):
    """Wraps a list of Tensor transformation functions to create a single
    function call which provides all necessary arguments for a forward hook.
    Can then be applied using `register_forward_hook(fn)`.
    """
    print("Weight modification not currently supported.")
    exit(1)

    # TODO
    # Need a different approach here: forward_hooks are called after computation
    # has finished, which is too late. Perhaps in register_forward_pre_hook,
    # store the module's weights under a different name and replace them with
    # a modified version. Probably change them back with register_forward_hook.
    
    assert len(functions) > 0
    
    def forward_hook(module, act_in, act_out):
        return _map_tensor(functions, act_out)
    
    return forward_hook


def _map_tensor(functions, tensors):
    """
    Apply the composition of all functions to all given tensors. If a tensor
    is None, it remains as None.
    :param functions: iterable collection of functions. Each must take a
    tensor and return a tensor of the same size. The first function is
    applied first.
    :param tensors: iterable collection of tensors.
    :return: tuple of tensors with identical shapes to input.
    """
    new_tensors = []

    for tensor in tensors:
        if tensor is None:
            new_tensors.append(None)
        else:
            for fn in functions:
                tensor = fn(tensor)
            new_tensors.append(tensor)

    return tuple(new_tensors)
