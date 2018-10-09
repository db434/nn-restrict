"""
Wrapper for torch.nn.functional which provides allows weights to be modified
before they are used.
"""

import torch


class QuantiseFunction(torch.autograd.Function):
    """
    Function with both a forward and a backward pass. I use a
    straight-through estimator here: the gradients are passed back as though
    no quantisation happened.
    """
    @staticmethod
    def forward(ctx, x, quantisation_fn):
        if quantisation_fn is not None:
            x = quantisation_fn(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Need to return gradients for all inputs of `forward`, including the
        # quantisation function.
        return grad_output, None


quantise = QuantiseFunction.apply
