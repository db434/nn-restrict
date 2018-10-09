import torch
import torch.nn as nn
import torch.nn.functional

from . import wrapped


class Shift(nn.Module):
    """Shift all feature maps by a fixed amount in different directions. Feature
    maps are split into a group for each kernel element and shifted so that the
    kernel element ends up in the centre of the kernel.
    
    Based on this paper: https://arxiv.org/abs/1711.08141"""

    def __init__(self, channels, kernel_size, dilation):
        super(Shift, self).__init__()

        # Ensure that kernel_size specifies both the width and height.
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        kernel_elements = self.kernel_size[0] * self.kernel_size[1]
        assert channels >= kernel_elements

        self.dilation = dilation
        self.group_size = channels // kernel_elements

        # Weird arguments needed for torch.nn.functional.pad.
        # Each pair of values represents the padding to each end of a dimension.
        # Starts from the final dimension (width) and works backwards.
        self.padding = ((self.kernel_size[1] // 2) * dilation,
                        (self.kernel_size[1] // 2) * dilation,
                        (self.kernel_size[0] // 2) * dilation,
                        (self.kernel_size[0] // 2) * dilation)

    def _extract_window(self, data, kx, ky, width, height):
        """Extract part of `data` corresponding to the given kernel position."""
        y_start = ky * self.dilation
        y_end = height - (self.kernel_size[0] - ky - 1) * self.dilation

        x_start = kx * self.dilation
        x_end = width - (self.kernel_size[1] - kx - 1) * self.dilation

        return data[:, :, y_start:y_end, x_start:x_end]

    def forward(self, x):
        # Add padding as though we're about to do a convolution.
        x = nn.functional.pad(x, self.padding)
        batch, channels, height, width = x.size()

        shifted = []

        # Slice a different window out of each group.
        group = 0
        for ky in range(self.kernel_size[0]):
            for kx in range(self.kernel_size[1]):
                chan_start = group * self.group_size
                chan_end = chan_start + self.group_size

                data = x[:, chan_start:chan_end, :, :]
                data = self._extract_window(data, kx, ky, width, height)

                shifted.append(data)

                group += 1

        # Ensure all channels are now in shifted.
        if group * self.group_size < channels:
            data = x[:, group * self.group_size:, :, :]
            data = self._extract_window(data, 0, 0, width, height)
            shifted.append(data)

        return torch.cat(shifted, dim=1)


# Same interface as torch.nn.Conv2d (except groups -> depth_multiplier).
class Conv2d(nn.Module):
    """A drop-in replacement for torch.nn.Conv2d which replaces non-trivial
    convolutions with a "shift" operation. This involves moving all pixels of
    a feature map by a fixed amount in a particular direction.
    
    Computation savings are proportional to kernel size squared.
    
    Based on this paper:
    https://arxiv.org/abs/1711.08141
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

        # Special cases:
        #  * If kernel_size = 1, there is no room to express a shift direction.
        if kernel_size == 1:
            print("INFO: using default convolution instead of shift.")
            print("  kernel_size = 1")
            self.conv = nn.Sequential(
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias),

                nn.BatchNorm2d(in_channels),
            )
        else:
            # There needs to be at least one channel shifted according to each
            # element of the "filter". Increase the number of intermediate
            # channels to accommodate this if necessary.
            intermediate_channels = max(out_channels, kernel_size ** 2)

            # ReLU and BN layers are in the position specified in the paper.
            # This is the CSC (conv-shift-conv) variant; there is also the
            # option to add an extra shift layer at the start.
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                # Pre-shift channel mixing.
                wrapped.Conv2d(in_channels=in_channels,
                               out_channels=intermediate_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=groups,
                               bias=False),

                # Shift (replaces KxK convolution).
                Shift(intermediate_channels, kernel_size, dilation),

                nn.BatchNorm2d(intermediate_channels),
                nn.ReLU(inplace=True),

                # Post-shift channel mixing.
                wrapped.Conv2d(in_channels=intermediate_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               dilation=1,
                               groups=groups,
                               bias=bias),
            )

    def forward(self, x):
        return self.conv(x)
