import random
import torch

from structured.depthwise_shuffle import *


def _test():
    """Test an entire depthwise-shuffle module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.
    """
    for i in range(100):
        inputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        outputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        padding = random.randint(1, 5)
        kernel_size = padding * 2 + 1  # ensure odd kernel size

        layer = Conv2d(inputs, outputs, kernel_size, padding=padding)
        in_data = torch.autograd.Variable(torch.Tensor(4, inputs, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)

    print("All tests passed.")


if __name__ == "__main__":
    _test()
