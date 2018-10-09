import torch.nn as nn

from . import wrapped


# Same interface as torch.nn.Conv2d (except groups -> depth_multiplier).
class Conv2d(nn.Module):
    """A drop-in replacement for torch.nn.Conv2d which uses a depthwise-
    separable structure. This means that a small number of filters are applied
    to each input channel, and then linear combinations of these intermediate
    results are taken to produce the output.
    
    In the limit of many channels, computations and weights are reduced in
    proportion to the filter size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=True):
        super(Conv2d, self).__init__()

        # Channel numbers can be scaled by floats, so need to be rounded back
        # to integers.
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Special case: if kernel_size = 1, factorising the convolution doesn't
        # add anything.
        if kernel_size == 1:
            print("INFO: using default convolution instead of separable.")
            print("  kernel_size = 1")
            self.conv = nn.Sequential(
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=1,
                               bias=bias),

                nn.BatchNorm2d(out_channels)
            )
        else:
            # This ordering of layers matches the MobileNet paper (assuming a
            # final ReLU is added in the higher-level network definition).
            # https://arxiv.org/abs/1704.04861
            self.conv = nn.Sequential(
                # Feature extraction. Each channel has `depth_multiplier`
                # different filters applied to it, each forming a separate
                # intermediate channel.
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=in_channels * depth_multiplier,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias),

                nn.BatchNorm2d(in_channels * depth_multiplier),
                nn.ReLU(inplace=True),

                # Cross-channel pooling.
                wrapped.Conv2d(in_channels=in_channels * depth_multiplier,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=bias),

                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x)
