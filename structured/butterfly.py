import torch
import torch.nn as nn
import torch.nn.functional

from . import wrapped


def _next_power_of_two(value):
    """Return the first power of 2 greater than or equal to the input."""
    power = 1
    while power < value:
        power *= 2
    return power


def _power_of_two(value):
    """Returns whether the given value is a power of two."""
    return (value & (value - 1)) == 0


def butterfly_sequence(inputs, outputs):
    """Return tuple of (inputs, outputs, butterfly size) for all sub-layers
    required to connect every input with every output.

    All sizes are powers of two, and the inputs and outputs may differ by a
    maximum factor of two at each stage.
    
    Note that the butterfly size may need to be scaled, depending on how these
    values are used. e.g. If inputs are duplicated in-place to match number of
    outputs, butterfly size must increase accordingly."""

    largest_butterfly = inputs
    smallest_butterfly = 2

    # Can only handle powers of 2. (Not much of a limitation.)
    assert _power_of_two(inputs) and _power_of_two(outputs)
    # Doesn't make sense to butterfly when there aren't enough inputs.
    assert inputs >= smallest_butterfly
    # There is a maximum rate at which the number of channels can increase.
    # Easy enough to get around, but keeping it simple for now.
    assert outputs <= inputs ** 2

    current_inputs = inputs
    current_butterfly = largest_butterfly

    # Go from largest butterfly to smallest. This is necessary because if there
    # are fewer outputs than inputs, there won't be space to use the largest
    # butterfly later.
    while current_butterfly >= smallest_butterfly:
        # Determine if we need to change the number of values to reach the
        # correct number of outputs.
        if outputs > current_inputs:
            current_outputs = current_inputs * 2
        elif outputs < current_inputs:
            current_outputs = current_inputs // 2
        else:
            current_outputs = current_inputs

        yield current_inputs, current_outputs, current_butterfly

        current_inputs = current_outputs
        current_butterfly //= 2


class Conv2dSublayer(nn.Module):
    """Class representing a single sublayer of the butterfly network. There will
    be log2(channels) of these sublayers, each with a different butterfly size.
    
    The typical case is:
     * Apply two filters to each input channel
     * Separate the intermediate result in two, one for each of the filters
     * Reorder one of the partitions according the the butterfly size
     * Add together the two partitions
    
    There are then minor modifications to this procedure if the number of
    outputs differs from the number of inputs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 butterfly_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Conv2dSublayer, self).__init__()

        assert _power_of_two(in_channels) and _power_of_two(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine how the number of channels changes as data passes through
        # this layer. This will affect the number of filters required.
        self.expansion = out_channels / in_channels

        # By default, apply two filters to each channel, one for each wing of
        # the butterfly.
        self.filters_per_channel = int(2 * self.expansion)

        self.intermediate_channels = self.in_channels * self.filters_per_channel

        self.butterfly_size = butterfly_size
        self.butterflies = (self.intermediate_channels // 2) // \
                           self.butterfly_size

        assert self.butterfly_size >= 2
        assert self.butterflies > 0 or self.expansion < 1

        # Apply multiple filters to each input channel, but don't combine the
        # results. This looks like the first phase of a depthwise-separable
        # convolution. The actual butterfly happens in forward().
        self.conv = wrapped.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.intermediate_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=self.in_channels,
                                   bias=False)

        self._weight_matrix_mask = None
        self._weight_matrix = None

    def extract_sequences(self, data):
        """Separate data along the channel dimension into two smaller datasets.
        Consecutive channels are assigned alternately to the two outputs."""
        batch, channels, height, width = data.size()

        # Simpler split if we didn't apply multiple filters to each input. Split
        # into first half of channels and second half.
        if self.filters_per_channel == 1:
            return data[:, :channels // 2, :, :].contiguous(), \
                   data[:, channels // 2:, :, :].contiguous()
        else:
            split = data.view(batch, channels // 2, 2, height, width)
            return split.select(2, 0).contiguous(), \
                split.select(2, 1).contiguous()

    def swap_wings(self, data):
        """Swap the wings of each butterfly.
    
        e.g. Butterfly size = 4
             Input  = 0 1 2 3   4 5 6 7
             Output = 2 3 0 1   6 7 4 5

        Method:
         1. Introduce dummy dimensions: (channels) -> (butterflies, 2, wing)
         2. Reflect middle dimension: [0,1] -> [1,0]
         3. Flatten dummy dimensions: (butterflies, 2, wing) -> (channels)
        """
        # If there was only one filter applied, then all of `data` is a single
        # wing, and no swapping is needed.
        if self.filters_per_channel == 1:
            return data

        batch, channels, height, width = data.size()

        wing_size = self.butterfly_size // 2
        assert wing_size * 2 == self.butterfly_size

        split = data.view(batch, self.butterflies, 2, wing_size, height, width)
        left = split[:, :, :1, :, :, :]
        right = split[:, :, 1:, :, :, :]
        merged = torch.cat([right, left], dim=2).contiguous()

        return merged.view(batch, channels, height, width)

    def _weight_matrix_rows(self, column):
        """Return the sequence of dense matrix rows which will be non-zero in
        this column.
        
        Weights come in pairs and are spaced butterfly_size // 2 apart, and 
        subsequent pairs are in_channels rows apart.
        
        The starting row increments by 1 for each column in the same wing,
        resets for the second wing of the same butterfly, and increments by
        butterfly_size for a new butterfly.
        """
        row = column % (self.butterfly_size // 2)
        row += (column // self.butterfly_size) * self.butterfly_size

        returned = 0

        while row < self.out_channels:
            assert returned < self.filters_per_channel
            yield row
            returned += 1

            if returned % 2 == 0:  # Start new pair of weights
                row += self.in_channels - (self.butterfly_size // 2)
            else:  # Continue current pair of weights
                row += self.butterfly_size // 2

    def weight_matrix_mask(self):
        """Return a ByteTensor showing which values in the dense weight matrix
        are non-zero, to be used with `torch.masked_scatter_`. The output of
        `torch.masked_scatter_` will need to be transposed since the scatter
        function does not allow control over which value goes to which position.
        """

        mask = torch.ByteTensor(self.in_channels, self.out_channels)
        mask.zero_()

        for column in range(self.in_channels):
            for row in self._weight_matrix_rows(column):
                mask[column][row] = 1

        in_channels, out_channels = mask.size()
        assert in_channels == self.in_channels
        assert out_channels == self.out_channels

        return mask

    def weight_matrix(self):
        """Convert the sparse weights stored internally to a dense
        representation which can be passed to an ordinary convolution routine.
        
        This will ultimately require more computation, but can perform better
        on a GPU.
        """

        sparse = list(self.conv.parameters())[0]

        # Initialise data buffers if this is the first time using them.
        if self._weight_matrix is None:
            self._weight_matrix_mask = self.weight_matrix_mask()
            self._weight_matrix = torch.zeros(self._weight_matrix_mask.size())

            if sparse.is_cuda:
                self._weight_matrix_mask = self._weight_matrix_mask.cuda()
                self._weight_matrix = self._weight_matrix.cuda()

        dense = self._weight_matrix.fill_(0)
        mask = self._weight_matrix_mask

        mask = torch.autograd.Variable(mask, requires_grad=False)
        dense = torch.autograd.Variable(dense)

        dense.masked_scatter_(mask, sparse)
        dense = dense.transpose(0, 1)  # Required by weight_matrix_mask()

        return dense

    def forward(self, x):
        x = self.conv(x)

        # Extract results from one filter application from each input channel.
        x1, x2 = self.extract_sequences(x)

        # Reorder data from the second filter applied to each channel so that
        # the wings of each butterfly are swapped.
        x2 = self.swap_wings(x2)

        # Combine the default ordered and reordered data.
        x = x1 + x2

        return x


class Conv2d(nn.Module):
    """A drop-in replacement for torch.nn.Conv2d, but using a butterfly
    connection structure internally.
    
    The cost of the butterfly is O(nlogn) compared with O(n^2) for an ordinary
    convolution layer, where n is the number of input channels.
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.norm = nn.BatchNorm2d(out_channels)

        # Special case if there are 2 inputs or less: there isn't space for a
        # whole butterfly, so use a normal convolution layer.
        if in_channels <= 2:
            print("INFO: using default convolution instead of butterfly.")
            print("  in_channels =", in_channels)
            self.conv = wrapped.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=1,
                                       bias=bias)
            self.expand_inputs = False
            self.trim_outputs = False
        else:
            butterflies = []

            self.expand_inputs = not _power_of_two(in_channels)
            self.trim_outputs = not _power_of_two(out_channels)

            start_channels = _next_power_of_two(in_channels)
            end_channels = _next_power_of_two(out_channels)

            for inputs, outputs, size in butterfly_sequence(start_channels,
                                                            end_channels):
                # When the number of outputs increases, we apply extra filters
                # to each input channel. This means the butterflies must be
                # larger than the default butterfly sequence would suggest.
                # The opposite does not happen when outputs decrease.
                expansion = max(1, outputs // start_channels)
                size *= expansion

                butterflies.append(Conv2dSublayer(in_channels=inputs,
                                                  out_channels=outputs,
                                                  kernel_size=kernel_size,
                                                  butterfly_size=size,
                                                  stride=stride,
                                                  padding=padding,
                                                  dilation=dilation,
                                                  groups=groups,
                                                  bias=bias))

                # Override some of the parameters so they don't have cumulative
                # effects.
                padding = kernel_size // 2
                stride = 1

                # Successive iterations with kernel_size > 1 also have a
                # cumulative effect, but this is a good effect, so I leave it.

            self.conv = nn.Sequential(*butterflies)

    def expand_to_power_of_two(self, x):
        """Expand x so it has a power-of-two number of channels. This is done by
        concatenating the data with itself, and then slicing it to the required
        size."""
        batch, channels, height, width = x.size()

        x = torch.cat([x, x], dim=1)
        return self.trim(x, _next_power_of_two(channels)).contiguous()

    @staticmethod
    def trim(x, num_channels):
        """Trim x so that it has the specified number of channels."""
        return x[:, :num_channels, :, :].contiguous()

    def can_use_fast_forward(self):
        """Determine whether the parameters of this layer allow the GPU-
        optimised computation to be used."""
        # The technique takes the same pixel from each channel as a vector, and
        # applies a transformation.
        #  * This doesn't work if kernel_size > 1 because then more pixels are
        #    needed.
        #  * I get out-of-memory errors if there are too many channels because
        #    the transformation matrix is size (in_channels)^2
        return self.kernel_size == 1 and self.in_channels < 10000

    def fast_forward(self, x):
        """Despite orders of magnitude less computation, I haven't found a way
        to execute a butterfly layer efficiently on a GPU.
        
        This function expands the weights out to the shape they would be in an
        ordinary convolution layer, then applies ordinary convolution."""
        assert self.can_use_fast_forward()

        weights = None

        for butterfly in self.conv.children():
            if weights is None:
                weights = butterfly.weight_matrix()
            else:
                weights = torch.matmul(butterfly.weight_matrix(), weights)

        # Dimensions must be (out_chans, in_chan, kernel_height, kernel_width).
        weights = weights.view(*weights.size(), 1, 1)

        # TODO: could potentially have the weight matrix handle odd input/output
        # sizes
        if self.expand_inputs:
            x = self.expand_to_power_of_two(x)

        x = nn.functional.conv2d(x, weights, stride=self.stride,
                                 padding=self.padding)

        if self.trim_outputs:
            x = self.trim(x, self.out_channels)

        x = self.norm(x)

        return x

    def forward(self, x):

        # TODO
        # Try to avoid breaking the test which checks that the outputs are the
        # same.

        if self.can_use_fast_forward() and x.is_cuda:
            return self.fast_forward(x)
        else:
            if self.expand_inputs:
                x = self.expand_to_power_of_two(x)

            x = self.conv(x)

            if self.trim_outputs:
                x = self.trim(x, self.out_channels)

            x = self.norm(x)

            return x
