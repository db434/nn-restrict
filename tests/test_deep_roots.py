import random
import torch

from structured.deep_roots import *


def _deep_roots_test():
    """Test an entire deep-roots module.

    For a range of different input and output sizes, ensure that the layer
    doesn't crash when running.
    """
    for i in range(100):
        inputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        outputs = random.randint(1, 10) * 8  # Ensure multiple of groups
        padding = random.randint(0, 5)
        kernel_size = padding * 2 + 1  # Ensure odd kernel size

        layer = Conv2d(inputs, outputs, kernel_size, padding=padding)
        in_data = torch.autograd.Variable(torch.Tensor(4, inputs, 10, 10))
        out_data = layer(in_data)
        assert out_data.size() == (4, outputs, 10, 10)

    print("All tests passed.")


if __name__ == "__main__":
    _deep_roots_test()
