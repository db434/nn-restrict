import random
import torch

from structured.butterfly import *


def _group_counts_to_group_sizes(params, group_counts):
    """Convert numbers of groups to sizes of groups."""
    _, out_channels, _ = params
    group_sizes = [out_channels // count for count in group_counts]
    return group_sizes


def _sequence_test():
    """Test that the sequence of butterflies is sensible."""

    # Each test case is a tuple of (parameters, group_sizes).
    # Parameters is itself a tuple of (in channels, out channels, min group).
    # Group sizes are measured with respect to the output channels.
    # The cost of a butterfly is proportional to the sum all group sizes (if
    # inputs == outputs).
    test_cases = [
        # Simple, same size all the way through.
        ((64, 64, 2), [2, 2, 2, 2, 2, 2]),
        ((64, 64, 4), [4, 4, 4]),

        # Varying group sizes (channels aren't powers of two).
        ((36, 36, 2), [2, 2, 3, 3]),
        ((36, 36, 3), [3, 3, 4]),
        ((36, 36, 4), [6, 6]),      # [4,9] also works but costs more (12 vs 13)

        # More outputs than inputs.
        # First butterfly connects groups of 2 inputs to 8 outputs.
        ((16, 64, 2), [8, 2, 2, 2]),
        ((24, 96, 2), [8, 2, 2, 3]),  # [12,2,2,2] is suboptimal

        # More inputs than outputs.
        # First butterflies connect groups of 2 or 4 inputs to only 1 output.
        ((64, 32, 2), [1, 2, 2, 2, 2, 2]),
        ((64, 16, 2), [1, 2, 2, 2, 2]),
        ((96, 24, 2), [1, 2, 2, 2, 3])
    ]

    for (params, correct) in test_cases:
        group_counts = Conv2d.get_group_counts(*params)
        group_sizes = _group_counts_to_group_sizes(params, group_counts)
        if group_sizes != correct:
            print("For {0} inputs, {1} outputs and min size {2}".format(
                *params))
            print("Expected", correct)
            print("Got     ", group_sizes)
            exit(1)


def _sublayer_test():
    """Test a single butterfly sub-layer.

    TODO: not really sure what can be tested here.
    Perhaps set all weights to 1 and check that inputs propagate through to
    only the expected outputs?
    """
    None


def _module_test():
    """Test an entire butterfly module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.

    Note: I don't yet test the output for correctness.
    """
    for i in range(100):
        inputs = random.randint(3, 100)  # Can't have butterflies smaller than 2
        outputs = random.randint(1, 100)
        kernel_size = 1
        min_butterfly_size = random.randint(2, 16)

        layer = Conv2d(inputs, outputs, kernel_size, min_butterfly_size)
        in_data = torch.Tensor(4, inputs, 10, 10).normal_(mean=0, std=1)
        in_data = torch.autograd.Variable(in_data)
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)


def _butterfly_test():
    """Test all components of a butterfly layer."""
    _sequence_test()
    _sublayer_test()
    _module_test()

    print("All tests passed.")


if __name__ == "__main__":
    _butterfly_test()
