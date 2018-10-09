from functools import reduce
import torch.nn as nn


class Conv2d(nn.Conv2d):
    """Simple wrapper for the default convolution class. 
    
    Adds a couple of extra methods to extract useful statistics. This module
    should not be visible outside of this package.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 **kwargs):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)

    def num_weights(self):
        """Returns the number of weights required to perform the computation."""

        num_weights = 0
        for parameters in self.parameters():
            num_weights += parameters.numel()

        return num_weights

    def num_operations(self, input_size, output_size):
        """Returns the number of operations required to complete this layer.
        1 multiply-accumulate = 1 operation.
        
        Both inputs are tuples of the form (batch, channels, height, width)."""

        convolutions = input_size[1] * output_size[1] // self.groups

        out_pixels = output_size[2] * output_size[3]
        kernel_pixels = reduce(lambda x, y: x * y, self.kernel_size)
        macs_per_convolution = out_pixels * kernel_pixels
        macs = macs_per_convolution * convolutions

        return macs
