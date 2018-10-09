"""
Module with various options for controlling number formats in the network.
"""

from collections.abc import Iterable
import torch.nn.functional

from .modules import *


def transform_gradients(model, transformation):
    """
    Apply a transformation to all gradient computations in the model.

    :param model: Model (or part of model) to apply the transformations on.
    :param transformation: Function which takes a tensor and returns a
    transformed tensor. A list of such functions is also accepted and will be
    applied in order.
    """

    # Ensure the interface of the functions matches the way we receive
    # gradients.
    fn = _wrap_for_gradients(transformation)

    # Only interested in bottom-level modules. Otherwise we might apply the
    # same transformation multiple times.
    # TODO: Check whether this quantises gradients of parameters. If not,
    # consider using Tensor.register_hook() on all parameters.
    leaves = (module for module in model.modules()
              if len(list(module.children())) == 0)

    for module in leaves:
        module.register_backward_hook(fn)


def transform_activations(model, transformation):
    """
    Apply a transformation to all activations in the model.

    :param model: Model (or part of model) to apply the transformations on.
    :param transformation: Function which takes a tensor and returns a
    transformed tensor. A list of such functions is also accepted and will be
    applied in order.
    """

    # Ensure the interface of the functions matches the way we receive
    # activations.
    fn = _wrap_for_activations(transformation)

    # Only interested in Quantiser modules.
    quantisers = (module for module in model.modules()
                  if isinstance(module, Quantiser))

    for module in quantisers:
        module.set_quantisation(fn)


def transform_weights(model, transformation):
    """
    Apply a transformation to all weights in the model.

    :param model: Model (or part of model) to apply the transformations on.
    :param transformation: Function which takes a tensor and returns a
    transformed tensor. A list of such functions is also accepted and will be
    applied in order.
    """

    # Ensure the interface of the functions matches the way we receive
    # weights.
    fn = _wrap_for_weights(transformation)

    # Only interested in modules which can accept a weight transform function.
    modules = (module for module in model.modules()
               if hasattr(module, "set_weight_transform"))

    for module in modules:
        module.set_weight_transform(fn)


def restrict_gradients(model, minimum=0.0, maximum=0.0, noise=0.0,
                       precision=0.0):
    """Place restrictions on possible gradient values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        transform_gradients(model, modifiers)


def restrict_activations(model, minimum=0.0, maximum=0.0, noise=0.0,
                         precision=0.0):
    """Place restrictions on possible activation values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        transform_activations(model, modifiers)


def restrict_weights(model, minimum=0.0, maximum=0.0, noise=0.0, precision=0.0):
    """Place restrictions on possible weight values for all layers of the
    model."""
    
    modifiers = _get_modifiers(minimum, maximum, noise, precision)
    
    if len(modifiers) > 0:
        transform_weights(model, modifiers)


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
    functions = _make_iterable(functions)

    def backward_hook(module, grad_in, grad_out):
        return _map_tensor(functions, grad_in)
    
    return backward_hook


def _wrap_for_activations(functions):
    """Wrap a list of Tensor transformation functions to create a single
    function call which takes a single tensor as its argument.
    """
    functions = _make_iterable(functions)
    
    def forward(tensor):
        for fn in functions:
            tensor = fn(tensor)
        return tensor

    return forward


def _wrap_for_weights(functions):
    """Wrap a list of Tensor transformation functions to create a single
    function call which takes a single tensor as its argument.
    """
    functions = _make_iterable(functions)

    def forward(tensor):
        for fn in functions:
            tensor = fn(tensor)
        return tensor

    return forward


def _make_iterable(data):
    """
    If a value is not already iterable, make it so.

    :param data: Arbitrary data.
    :return: Iterable object containing `data`.
    """
    if not isinstance(data, Iterable):
        data = [data]

    return data


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
