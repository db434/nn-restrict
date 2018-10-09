import torch.nn as nn

from . import wrapped


# As introduced in:
#
# Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups
# https://arxiv.org/abs/1605.06489

class Conv2d(nn.Module):
    """A drop-in replacement for torch.nn.Conv2d which uses the Deep Roots
    structure. This involves using a grouped convolution to produce intermediate
    channels, and then using linear combinations of these to produce the output.
    
    The grouped convolution is `groups` times cheaper, and the linear
    combination is `filter size` times cheaper than ordinary convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=8,  # Sensible default
                 bias=True):
        super(Conv2d, self).__init__()

        # Channel numbers can be scaled by floats, so need to be rounded back
        # to integers.
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Degenerate into normal conv layer if there are too few channels.
        # Could instead reduce number of groups to fit, but I think this is
        # simpler.
        if in_channels < groups or out_channels < groups:
            print("INFO: using default convolution instead of deep roots.")
            print("  Inputs:", in_channels, ", outputs:", out_channels,
                  ", groups:", groups)

            self.conv = nn.Sequential(
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=1,
                               bias=bias),

                nn.BatchNorm2d(out_channels),
            )
        else:
            # The paper mentions that batch normalisation is used, but it 
            # doesn't seem to say where. I follow MobileNet and put it after
            # each convolution, but before the activation function.
            self.conv = nn.Sequential(
                # Grouped convolution.
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias),

                nn.BatchNorm2d(out_channels),

                nn.ReLU(inplace=True),

                # Channel mixing.
                wrapped.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=bias),

                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)
