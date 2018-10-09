import torch.nn as nn

from . import wrapped
from util import log


# A simplification of the module used in:
# 
# ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
# Devices
# https://arxiv.org/abs/1707.01083
#
# This implementation simply performs a grouped convolution, then permutes the
# channels. The permutation takes the first channel from each group, then the
# second from each group, etc.

class Conv2d(nn.Module):
    """A replacement for torch.nn.Conv2d, but using a group shuffle structure.
    
    This module is not strictly a drop-in replacement because not every input is
    able to influence every output. Multiple layers must be stacked together to
    achieve that property.
    
    Computation and memory requirements are reduced by a factor of `groups`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=8,  # Sensible default from paper (ImageNet models)
                 bias=True,
                 **kwargs):
        super(Conv2d, self).__init__()

        # Channel numbers can be scaled by floats, so need to be rounded back
        # to integers.
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Degenerate into normal conv layer if there are too few channels.
        # Could instead reduce number of groups to fit, but I think this is
        # simpler.
        if in_channels < groups or out_channels < groups or \
                in_channels % groups != 0 or out_channels % groups != 0:
            log.info("INFO: using default convolution instead of shuffle.")
            log.info("  Inputs:", in_channels, ", outputs:", out_channels,
                     ", groups:", groups)

            self.groups = 1
            self.conv = nn.Sequential(
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=1,
                               bias=bias,
                               **kwargs),

                nn.BatchNorm2d(out_channels)
            )
        else:
            # Put the batch-norm after the convolution to match depthwise-
            # separable, for which we have a reference specifying where it
            # should go.
            self.groups = groups
            self.conv = nn.Sequential(
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               **kwargs),

                nn.BatchNorm2d(out_channels)
            )

    @staticmethod
    def shuffle(x, groups):
        if groups == 1:
            return x

        # Uniform shuffle of channels. Assumes a particular dimension order.      
        batch, channels, height, width = x.size()

        x = x.view(batch, groups, channels // groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)

    def forward(self, x):
        output = self.conv(x)

        output = self.shuffle(output, self.groups)

        return output
