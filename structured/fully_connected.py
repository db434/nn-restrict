import torch.nn as nn

from . import wrapped


class Conv2d(nn.Module):
    """Simple wrapper for the default convolution class. Introduced so all
    convolution variants have a similar interface.
    
    Adds batch normalisation to be fair, since the other deeper modules require
    it.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv2d, self).__init__()

        # Channel numbers can be scaled by floats, so need to be rounded back
        # to integers.
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Put the batch-norm after the convolution to match depthwise-separable,
        # for which we have a reference specifying where it should go.
        self.conv = nn.Sequential(
            wrapped.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias),

            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)
