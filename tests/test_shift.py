import random

from structured.shift import *


def _functional_test():
    """Ensure that data is being shifted properly."""
    # TODO
    None


def _shift_test():
    """Test creating and running Shift modules with a variety of configurations.
    """
    for i in range(100):
        kernel_size = random.randint(1, 5) * 2 + 1  # ensure odd kernel size
        channels = kernel_size ** 2 + random.randint(1, 100)
        dilation = random.randint(1, 4)

        layer = Shift(channels, kernel_size, dilation)
        in_data = torch.autograd.Variable(torch.Tensor(4, channels, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, channels, 10, 10)


def _module_test():
    """Test an entire shift convolution module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.
    """
    for i in range(100):
        inputs = random.randint(1, 100)
        outputs = random.randint(1, 100)
        padding = random.randint(1, 5)
        kernel_size = padding * 2 + 1  # ensure odd kernel size

        layer = Conv2d(inputs, outputs, kernel_size, padding=padding)
        in_data = torch.autograd.Variable(torch.Tensor(4, inputs, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)


if __name__ == "__main__":
    _functional_test()
    _shift_test()
    _module_test()

    print("All tests passed.")
