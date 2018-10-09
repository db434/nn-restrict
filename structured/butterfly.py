from functools import reduce
import math
import torch.nn as nn

from . import wrapped
from util import log


class Conv2dSublayer(nn.Module):
    """Class representing a single sublayer of the butterfly network.

    This module performs a single grouped convolution and then reorders the
    channels. Channels are placed together if they have so far received
    contributions from different subsets of input channels. This means that
    in the next sublayer, they will all contribute to each other."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups,
                 input_cone,  # Number of original channels reaching each output
                 **kwargs):
        super(Conv2dSublayer, self).__init__()

        self.conv = wrapped.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   groups=groups,
                                   bias=False,
                                   **kwargs)

        assert out_channels % input_cone == 0
        self.shuffle_groups = out_channels // input_cone

    def shuffle(self, x):
        groups = self.shuffle_groups
        if groups == 1:
            return x

        # Uniform shuffle of channels. Assumes a particular dimension order.
        batch, channels, height, width = x.size()

        x = x.view(batch, groups, channels // groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
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
                 bias=True,
                 args=None):
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

        # Compute the sequence of butterflies to be used.
        # TODO: if the number of channels is prime (or nearly prime),
        # we might want to add dummy channels to give a better butterfly
        # sequence. More factors = smaller factors = less computation.
        group_counts = self.get_group_counts(in_channels, out_channels,
                                             args.min_bfly_size)

        # Special case: if we were unable to generate a valid sequence of
        # butterflies, use a normal convolution.
        if len(group_counts) == 0:
            log.info("INFO: using default convolution instead of butterfly.")
            log.info("  in_channels =", in_channels)
            log.info("  out_channels =", out_channels)
            log.info("  min_butterfly_size =", args.min_bfly_size)

            self.conv = wrapped.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=1,
                                       bias=bias,
                                       args=args)
        else:
            butterflies = []

            # Number of consecutive channels which have all been computed
            # using the same subset of inputs.
            silo_size = 1

            # The number of input channels may vary as we move through the
            # sublayers.
            current_channels = in_channels

            for group_count in group_counts:
                assert current_channels % group_count == 0
                assert out_channels % group_count == 0

                out_group_size = out_channels // group_count
                silo_size *= out_group_size

                butterflies.append(Conv2dSublayer(in_channels=current_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  groups=group_count,
                                                  input_cone=silo_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  dilation=dilation))

                current_channels = out_channels

                # Override some of the parameters so they don't have cumulative
                # effects.
                padding = kernel_size // 2
                stride = 1

                # Successive iterations with kernel_size > 1 also have a
                # cumulative effect, but this is a good effect, so I leave it.

            self.conv = nn.Sequential(*butterflies)

    @staticmethod
    def get_group_counts(in_channels, out_channels, min_group_size):
        """Determine which convolution group counts should be used for the
        given number of channels. Each group count will be used to generate a
        separate grouped convolution sub-layer."""

        # The first grouped convolution allows groups of X output channels to
        # share the same inputs. After the appropriate rearrangement of
        # channels, the next convolution connects Y groups, so XY channels now
        # share the same inputs. Continue until all output channels share all
        # the inputs.
        #
        # Assertion 1: in_channels changes as we add sublayers, and ends up
        # equal to out_channels.
        #
        # Assertion 2: every group count used is a factor of both the
        # input channels and output channels for that sublayer.
        #
        # Assertion 3: the product of all group sizes is equal to out_channels.

        if in_channels < min_group_size or out_channels < min_group_size:
            return []

        group_sizes = []
        group_counts = []

        # If we need to change the number of channels, add a one-off group count
        # which is only made from shared factors. Find the smallest group
        # size (largest group count) which satisfies the min_group_size.
        initial_group_count = Conv2d._best_initial_group_count(
            in_channels, out_channels, min_group_size)
        if initial_group_count > 0:
            group_counts.append(initial_group_count)
            group_sizes.append(out_channels // initial_group_count)
        # At this point, assertion 1 should hold.

        # Define a silo to be a group of channels which share the same
        # *original* inputs. We aim to connect all silos by the end of the
        # layer.
        if len(group_sizes) > 0:
            num_silos = out_channels // group_sizes[0]
        else:
            num_silos = out_channels

        remaining_group_sizes = Conv2d._best_group_sizes(num_silos,
                                                         min_group_size)
        group_sizes += remaining_group_sizes
        for group_size in remaining_group_sizes:
            group_counts.append(out_channels // group_size)

        assert len(group_sizes) > 0
        assert len(group_counts) > 0

        # Check that assertion 2 holds. Uses the fact that in_channels ==
        # out_channels for all sublayers except possibly the first one.
        assert in_channels % group_counts[0] == 0
        for group_count in group_counts:
            assert out_channels % group_count == 0

        # Check that assertion 3 holds.
        assert _list_product(group_sizes) == out_channels

        return group_counts

    @staticmethod
    def _best_initial_group_count(in_channels, out_channels, min_group_size):
        """Determine the best butterfly group count to translate between
        the given numbers of input and output channels."""

        # Special case: no translation needed.
        if in_channels == out_channels:
            return 0

        # Need group sizes which divide perfectly into both in_channels and
        # out_channels. Find all common factors.
        in_factors = _prime_factorisation(in_channels)
        out_factors = _prime_factorisation(out_channels)
        intersection = _list_intersection(in_factors, out_factors)

        # Want the largest group count allowed, so sort in reverse order.
        compound_factors = list(reversed(sorted(_compound_factors(
            intersection))))

        # No common factor: can't use grouped convolution.
        if len(compound_factors) == 0:
            return 1

        for group_count in compound_factors:
            if in_channels // group_count >= min_group_size:
                return group_count

        # If there were no valid factors, do the best we can do.
        return compound_factors[-1]

    @staticmethod
    def _best_group_sizes(channels, min_group_size):
        """Determine the best (lowest cost) butterfly sequence which connects
        the given number of inputs and outputs. Assumes the same number of
        inputs and outputs."""

        # Total cost is is channels * group size, but all sequences generated
        # will share the same `channels` value so that is omitted here.
        # TODO: Include a preference for shorter sequences if there's a tie.
        def cost(sequence):
            return _list_sum(sequence)

        candidates = _all_factorisations(channels, min_group_size)

        best_sequence = []
        best_cost = math.inf

        for c in candidates:
            sequence_cost = cost(c)
            if sequence_cost < best_cost:
                best_cost = sequence_cost
                best_sequence = c

        return best_sequence

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def _prime_factorisation(value):
    """Return a list of prime factors, in order from smallest to largest."""
    factors = []
    factor = 2
    remaining = value

    while factor <= remaining:
        if remaining % factor == 0:
            factors.append(factor)
            remaining //= factor
        else:
            factor += 1

    assert _list_product(factors) == value
    return factors


def _all_factorisations(value, minimum_factor):
    """Return a list of lists containing all possible factorisations of
    `value`. At most one factor may be less than the minimum."""
    factorisations = []

    factors = [x for x in range(2, value+1) if value % x == 0]
    for factor in factors:
        remainder = value // factor
        if remainder == 1:
            factorisations.append([factor])
        else:
            for factorisation in _all_factorisations(remainder, minimum_factor):
                factorisation.append(factor)
                factorisations.append(factorisation)

    # Remove duplicates. Would like to use a set, but they don't allow lists.
    factorisations = [sorted(f) for f in factorisations]
    no_duplicates = []
    for f in factorisations:
        if f not in no_duplicates:
            no_duplicates.append(f)

    # Filter to remove factorisations which have more than 1 factor below the
    # minimum.
    filtered = []
    for f in no_duplicates:
        too_small = 0
        for factor in f:
            if factor < minimum_factor:
                too_small += 1
        if too_small <= 1:
            filtered.append(f)

    return filtered


def _compound_factors(prime_factors):
    """Return a set of all compound factors, given the list of prime factors."""
    compound = set()

    for position, factor in enumerate(prime_factors):
        compound.add(factor)
        remaining = _compound_factors(prime_factors[position+1:])
        for value in remaining:
            compound.add(factor * value)
            compound.add(value)

    return compound


def _list_intersection(list1, list2):
    """Compute the list of all elements present in both list1 and list2.
    Duplicates are allowed. Assumes both lists are sorted."""
    intersection = []

    pos1 = 0
    pos2 = 0

    while pos1 < len(list1) and pos2 < len(list2):
        val1 = list1[pos1]
        val2 = list2[pos2]

        if val1 == val2:
            intersection.append(val1)
            pos1 += 1
            pos2 += 1
        elif val1 < val2:
            pos1 += 1
        else:
            pos2 += 1

    return intersection


def _list_difference(list1, list2):
    """Compute the list of all elements present in list1 but not list2.
    Duplicates are allowed. Assumes both lists are sorted."""
    difference = []

    pos1 = 0
    pos2 = 0

    while pos1 < len(list1) and pos2 < len(list2):
        val1 = list1[pos1]
        val2 = list2[pos2]

        if val1 == val2:
            pos1 += 1
            pos2 += 1
        elif val1 < val2:
            difference.append(val1)
            pos1 += 1
        else:
            pos2 += 1

    difference += list1[pos1:]

    return difference


def _list_sum(l):
    """Compute the sum of all elements of the list."""
    return reduce(lambda x, y: x+y, l, 0)


def _list_product(l):
    """Compute the product of all elements of the list."""
    return reduce(lambda x, y: x*y, l, 1)
