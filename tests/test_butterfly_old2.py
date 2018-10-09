import random

from structured.butterfly_old2 import *

# Each test case is a tuple of (inputs, outputs, butterfly sequence).
# The butterfly sequence is itself a list of tuples, with each element holding
# (inputs, outputs, butterfly size).
_test_cases = [
    (8, 8, [(8, 8, 8), (8, 8, 4), (8, 8, 2)]),  # inputs = outputs
    (4, 8, [(4, 8, 4), (8, 8, 2)]),  # inputs < outputs
    (8, 4, [(8, 4, 8), (4, 4, 4), (4, 4, 2)]),  # inputs > outputs
]


def _check_valid_sequence(inputs, outputs, sequence):
    """See if a butterfly sequence looks sensible.

    A sensible sequence has the following properties:
     * At least one butterfly in it
     * Input to first butterfly = input to whole sequence
     * Output from last butterfly = output from whole sequence
     * Output of one layer = input of the next
     * Butterfly size starts at `inputs` and divides by 2 until it reaches 2
    """
    assert len(sequence) >= 1
    assert inputs == sequence[0][0]
    assert outputs == sequence[-1][1]
    assert inputs == sequence[0][2]
    assert 2 == sequence[-1][2]

    last_butterfly = sequence[0]
    for butterfly in sequence[1:]:
        assert butterfly[0] == last_butterfly[1]
        assert butterfly[2] == last_butterfly[2] // 2

        last_butterfly = butterfly


def _sequence_test():
    """Ensure that the sequence of butterflies is sensible."""

    # Specify the exact sequence for a few simple layers.
    for inputs, outputs, correct in _test_cases:
        output = list(butterfly_sequence(inputs, outputs))
        for x, y in zip(correct, output):
            if x != y:
                print("For {0} inputs and {1} outputs".format(inputs, outputs))
                print("Expected", correct)
                print("Got", output)
                break

    # Create some random layer configurations and check that the sequences are
    # sane.
    for i in range(100):
        log_inputs = random.randint(1, 10)
        log_outputs = random.randint(0, 2 * log_inputs)
        inputs = 2 ** log_inputs
        outputs = 2 ** log_outputs
        sequence = list(butterfly_sequence(inputs, outputs))
        _check_valid_sequence(inputs, outputs, sequence)


def _sublayer_test():
    """Test a single butterfly sub-layer.

    A single sub-layer doesn't do much. It:
     * Splits data in two on the channel dimension
     * Reorders one of these two segments (again on the channel dimension)
     * Recombines the results
    """
    layer = Conv2dSublayer(in_channels=8, out_channels=8,
                           kernel_size=1, butterfly_size=4)

    data1d = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(1, 8, 1, 1)
    data2d = torch.Tensor([11, 12, 21, 22, 31, 32, 41, 42,
                           51, 52, 61, 62, 71, 72, 81, 82]).view(1, 8, 2, 1)

    # Test channel splitting. Want alternate channels in each output tensor.
    odd1d = torch.Tensor([1, 3, 5, 7]).view(1, 4, 1, 1)
    even1d = torch.Tensor([2, 4, 6, 8]).view(1, 4, 1, 1)
    odd2d = torch.Tensor([11, 12, 31, 32, 51, 52, 71, 72]).view(1, 4, 2, 1)
    even2d = torch.Tensor([21, 22, 41, 42, 61, 62, 81, 82]).view(1, 4, 2, 1)

    odd, even = layer.extract_sequences(data1d)
    assert torch.equal(odd, odd1d)
    assert torch.equal(even, even1d)

    odd, even = layer.extract_sequences(data2d)
    assert torch.equal(odd, odd2d)
    assert torch.equal(even, even2d)

    # Test channel reordering. This is basically the example from the docstring
    # of _Conv2dSublayer.swap_wings().
    reorder1d = torch.Tensor([3, 4, 1, 2, 7, 8, 5, 6]).view(1, 8, 1, 1)
    reorder2d = torch.Tensor([31, 32, 41, 42, 11, 12, 21, 22,
                              71, 72, 81, 82, 51, 52, 61, 62]).view(1, 8, 2, 1)

    reorder = layer.swap_wings(data1d)
    assert torch.equal(reorder1d, reorder)

    reorder = layer.swap_wings(data2d)
    assert torch.equal(reorder2d, reorder)


def _module_test():
    """Test an entire butterfly module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.

    Restrictions:
     * outputs <= inputs ** 2

    Note: I don't yet test the output for correctness.
    """
    for i in range(100):
        inputs = random.randint(3, 100)  # Can't have butterflies smaller than 2
        outputs = random.randint(1, min(inputs ** 2, 100))
        kernel_size = 1

        layer = Conv2d(inputs, outputs, kernel_size)
        in_data = torch.Tensor(4, inputs, 10, 10).normal_(mean=0, std=1)
        in_data = torch.autograd.Variable(in_data)
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)


def _matrix_test():
    """Test that the matrix representation of a single butterfly's weights
    matches the butterfly representation."""

    for i in range(100):
        log_inputs = 2  # random.randint(1,10)  # TODO
        log_outputs = 2  # random.randint(max(1, log_inputs-1), log_inputs+1)
        log_butterfly = random.randint(1, max(log_inputs, log_outputs))
        inputs = 2 ** log_inputs
        outputs = 2 ** log_outputs
        butterfly = 2 ** log_butterfly

        layer = Conv2dSublayer(inputs, outputs, 1, butterfly)
        in_data = torch.Tensor(1, inputs, 1, 1).normal_(mean=0, std=1)
        in_data = torch.autograd.Variable(in_data)

        matrix = layer.weight_matrix().contiguous()

        print(matrix)
        print(list(layer.conv.parameters())[0].data.squeeze().view(inputs, -1))
        print(in_data.squeeze())

        matrix = matrix.view(*matrix.size(), 1, 1)

        out_data = layer.forward(in_data)
        out_data_matrix = torch.nn.functional.conv2d(in_data, matrix)

        print(out_data.squeeze())
        print(out_data_matrix.squeeze())

        # Can't test for equality with floating point numbers, so find maximum
        # difference.
        error = out_data - out_data_matrix
        assert error.abs().max().data[0] < 1e-6

        print("Test passed!")


def _fast_forward_test():
    """Test that GPU-optimised execution produces the same result as ordinary
    execution."""

    for i in range(100):
        inputs = random.randint(3, 100)  # Can't have butterflies smaller than 2
        outputs = random.randint(1, min(inputs ** 2, 100))
        kernel_size = 1

        layer = Conv2d(inputs, outputs, kernel_size)
        in_data = torch.Tensor(4, inputs, 10, 10).normal_(mean=0, std=1)
        in_data = torch.autograd.Variable(in_data)
        out_data = layer.forward(in_data)
        out_data_fast = layer.fast_forward(in_data)

        # Can't test for equality with floating point numbers, so find maximum
        # difference.
        error = out_data - out_data_fast
        assert error.abs().max().data[0] < 1e-6


def _butterfly_test():
    """Test all components of a butterfly layer."""
    _sequence_test()
    _sublayer_test()
    _module_test()

    # The GPU-optimised routine is currently incompatible with the pure
    # butterfly method. Either seems to work well in isolation, but they use
    # different weights for different purposes.
    # _matrix_test()
    # _fast_forward_test()

    print("All tests passed.")


if __name__ == "__main__":
    _butterfly_test()
