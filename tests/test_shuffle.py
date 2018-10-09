import random
import torch

from structured.shuffle import *


def _shuffle_test_run(in_data, groups, out_data):
    """Wrapper for a single shuffle."""
    _, channels, _, _ = in_data.size()

    shuffler = Conv2d(channels, channels, 1, groups=groups)
    result = shuffler.shuffle(in_data, groups)

    assert torch.equal(result, out_data)


def _shuffle_test():
    """Try out a few simple shuffles to make sure things work as expected."""

    # Simple 1D data to start. Need to convert to Torch's format, and make it
    # look like it's 4D neural network data. Each value is a channel.
    data = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(1, 8, 1, 1)

    # Data shuffled with various numbers of groups.
    data1 = data
    data2 = torch.Tensor([1, 5, 2, 6, 3, 7, 4, 8]).view(1, 8, 1, 1)
    data4 = torch.Tensor([1, 3, 5, 7, 2, 4, 6, 8]).view(1, 8, 1, 1)
    data8 = data

    _shuffle_test_run(data, 1, data1)
    _shuffle_test_run(data, 2, data2)
    _shuffle_test_run(data, 4, data4)
    _shuffle_test_run(data, 8, data8)

    # 2D data. Each inner list is a separate channel.
    data = torch.Tensor([[11, 12], [21, 22], [31, 32], [41, 42]]).view(1, 4, 2,
                                                                       1)

    data1 = data
    data2 = torch.Tensor([[11, 12], [31, 32], [21, 22], [41, 42]]).view(1, 4, 2,
                                                                        1)
    data4 = data

    _shuffle_test_run(data, 1, data1)
    _shuffle_test_run(data, 2, data2)
    _shuffle_test_run(data, 4, data4)

    # 2D data. Each inner list is all the channels of a single batch.
    data = torch.Tensor([[11, 12, 13, 14], [21, 22, 23, 24]]).view(2, 4, 1, 1)

    data1 = data
    data2 = torch.Tensor([[11, 13, 12, 14], [21, 23, 22, 24]]).view(2, 4, 1, 1)
    data4 = data

    _shuffle_test_run(data, 1, data1)
    _shuffle_test_run(data, 2, data2)
    _shuffle_test_run(data, 4, data4)


def _module_test():
    """Construct layers with a range of different parameters and check that
    nothing crashes."""

    for i in range(100):
        inputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        outputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        padding = random.randint(0, 5)
        kernel_size = padding * 2 + 1  # Ensure odd kernel size

        layer = Conv2d(inputs, outputs, kernel_size, padding=padding)
        in_data = torch.autograd.Variable(torch.Tensor(4, inputs, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)


if __name__ == "__main__":
    _shuffle_test()
    _module_test()

    print("All tests passed.")
