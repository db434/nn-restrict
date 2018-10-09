import random

from structured.hadamard import *


def _sublayer_test():
    """Test a single Hadamard sub-layer.

    A single sub-layer doesn't do much. It:
     * Splits data in two on the channel dimension
     * Reorders one of these two segments (again on the channel dimension)
     * Recombines the results
    """
    layer = HadamardLayer(channels=8, butterfly_size=4)

    data1d = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(1, 8, 1, 1)
    data2d = torch.Tensor(
        [11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62, 71, 72, 81, 82]).view(
        1, 8, 2, 1)

    # Test channel splitting. Want alternating pairs of channels in each output
    # tensor. (Butterfly size = 4, so wings are 2 channels each.)
    odd1d = torch.Tensor([1, 2, 5, 6]).view(1, 4, 1, 1)
    even1d = torch.Tensor([3, 4, 7, 8]).view(1, 4, 1, 1)
    odd2d = torch.Tensor([11, 12, 21, 22, 51, 52, 61, 62]).view(1, 4, 2, 1)
    even2d = torch.Tensor([31, 32, 41, 42, 71, 72, 81, 82]).view(1, 4, 2, 1)

    odd, even = layer.extract_wings(data1d)
    assert torch.equal(odd, odd1d)
    assert torch.equal(even, even1d)

    odd, even = layer.extract_wings(data2d)
    assert torch.equal(odd, odd2d)
    assert torch.equal(even, even2d)

    # assemble_wings should do the reverse of extract_wings.
    assembled1d = layer.assemble_wings(odd1d, even1d)
    assert torch.equal(assembled1d, data1d)

    assembled2d = layer.assemble_wings(odd2d, even2d)
    assert torch.equal(assembled2d, data2d)


def _hadamard_func_test():
    """Ensure that the thing being computed is indeed the Hadamard transform."""

    # Using the example from here:
    # https://en.wikipedia.org/wiki/Hadamard_transform

    hadamard = Hadamard(channels=8)

    input1d = torch.Tensor([1, 0, 1, 0, 0, 1, 1, 0]).view(1, 8, 1, 1)
    output1d = torch.Tensor([4, 2, 0, -2, 0, 2, 0, 2]).view(1, 8, 1, 1)
    output = hadamard(input1d)
    assert torch.equal(output, output1d)

    input2d = torch.Tensor(
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]).view(1, 8, 2, 1)
    output2d = torch.Tensor(
        [4, 4, 2, 2, 0, 0, -2, -2, 0, 0, 2, 2, 0, 0, 2, 2]).view(1, 8, 2, 1)
    output = hadamard(input2d)
    assert torch.equal(output, output2d)


def _module_test():
    """Test an entire Hadamard module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.

    Restrictions:
     * outputs == inputs == a power of 2

    Note: I don't yet test the output for correctness.
    """
    for i in range(100):
        log_channels = random.randint(1, 10)
        channels = 2 ** log_channels
        kernel_size = 3

        layer = Conv2d(channels, channels, kernel_size, padding=1)
        in_data = torch.autograd.Variable(torch.Tensor(4, channels, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, channels, 10, 10)


def _hadamard_test():
    """Test all components of a Hadamard layer."""
    _sublayer_test()
    _hadamard_func_test()
    _module_test()

    print("All tests passed.")


if __name__ == "__main__":
    _hadamard_test()
