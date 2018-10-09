import torch
import torch.nn as nn

from . import wrapped
from util import log


def _power_of_two(value):
    """Returns whether the given value is a power of two."""
    return (value & (value - 1)) == 0


class HadamardLayer(nn.Module):
    """Split each butterfly in two, in1 and in2. For each butterfly:
    
    Output (out1, out2) = (in1 + in2, in1 - in2)
    """

    def __init__(self, channels, butterfly_size):
        super(HadamardLayer, self).__init__()

        assert _power_of_two(channels)
        assert _power_of_two(butterfly_size)

        self.channels = channels
        self.butterfly_size = butterfly_size
        self.butterflies = channels // butterfly_size

    def extract_wings(self, data):
        """Split `data` on the channel dimension into two tensors. Each tensor
        has alternating blocks of `butterfly_size//2` channels."""
        wing_size = self.butterfly_size // 2
        batch, channels, height, width = data.size()

        split = data.view(batch, self.butterflies, 2, wing_size, height, width)
        left = split[:, :, :1, :, :, :].contiguous()
        right = split[:, :, 1:, :, :, :].contiguous()
        left = left.view(batch, channels // 2, height, width)
        right = right.view(batch, channels // 2, height, width)

        return left, right

    def assemble_wings(self, wings1, wings2):
        """The reverse of `extract_wings`: take two tensors and merge them into
        one, with blocks of `butterfly_size//2` channels coming from alternate
        tensors."""
        wing_size = self.butterfly_size // 2
        batch, channels, height, width = wings1.size()

        left = wings1.view(batch, self.butterflies, 1, wing_size, height, width)
        right = wings2.view(batch, self.butterflies, 1, wing_size, height,
                            width)

        result = torch.cat([left, right], dim=2).contiguous()
        return result.view(batch, channels * 2, height, width)

    def forward(self, x):
        # Separate wings.
        in1, in2 = self.extract_wings(x)

        # Main computation.
        out1, out2 = in1 + in2, in1 - in2

        # Put wings back together (currently have all out1s then all out2s, but
        # want out1, out2, out1, out2, etc.).
        return self.assemble_wings(out1, out2)


class Hadamard(nn.Module):
    """Perform a Hadamard transform across the channels of the input data.    
    https://en.wikipedia.org/wiki/Hadamard_transform
    
    This means that effectively, a separate identical transform is being applied
    to every x,y position in the input.
    """

    def __init__(self, channels):
        super(Hadamard, self).__init__()

        butterfly_size = channels
        layers = []

        while butterfly_size >= 2:
            layers.append(HadamardLayer(channels, butterfly_size))
            butterfly_size //= 2

        self.mix = nn.Sequential(*layers)

    def forward(self, x):
        return self.mix(x)


class ChannelMixer(nn.Module):
    """Module which combines data from X input channels to produce X output
    channels. Uses the Hadamard transform where possible, and falls back to
    a 1x1 convolution otherwise.
    
    Hadamard transforms require input channels == output channels == power of 2.
    """

    def __init__(self, channels):
        super(ChannelMixer, self).__init__()

        # The Hadamard transform requires powers of two.
        self.can_use_hadamard = _power_of_two(channels)
        self.channels = channels

        if not self.can_use_hadamard:
            log.info("INFO: using 1x1 convolution instead of Hadamard "
                    "transform.")
            log.info("  channels =", channels)
            self.mix = nn.Sequential(
                wrapped.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            # One scaling factor per channel.
            # TODO: pre-divide the scaling factors by sqrt(channels)?
            self.scales = nn.Parameter(torch.randn(1, channels, 1, 1))
            self.mix = Hadamard(channels)

    def forward(self, x):
        if self.can_use_hadamard:
            scaled = x * self.scales
            return self.mix(scaled)
        else:
            return self.mix(x)


# Same interface as torch.nn.Conv2d (except groups -> depth_multiplier).
class Conv2d(nn.Module):
    """A drop-in replacement for torch.nn.Conv2d. A small number of filters are
    applied to each input channel, and then a Hadamard transform is applied to
    the results to produce the output.
    https://en.wikipedia.org/wiki/Hadamard_transform
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=True,
                 **kwargs):
        super(Conv2d, self).__init__()

        # Channel numbers can be scaled by floats, so need to be rounded back
        # to integers.
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Special case: if kernel_size = 1, there are no features to extract,
        # so just mix the channels.
        if kernel_size == 1 and in_channels == out_channels:
            self.conv = ChannelMixer(out_channels)
        else:
            # Update depth_multiplier if necessary so there are the same number
            # of intermediate channels as output channels.
            if in_channels * depth_multiplier != out_channels:
                depth_multiplier = out_channels // in_channels
                assert in_channels * depth_multiplier == out_channels

            self.conv = nn.Sequential(
                # Feature extraction.
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=in_channels * depth_multiplier,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=False,
                               **kwargs),

                nn.BatchNorm2d(in_channels),

                # Mix channels.
                ChannelMixer(out_channels)
            )

    def forward(self, x):
        return self.conv(x)
