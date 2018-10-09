from modifiers.numbers import *


def _check_equal(input_data, modifier, expected_result):
    """
    Verify that modifier(input_tensor) == expected_result. Throws an assertion
    error if not.
    :param input_data: array of arbitrary dimensions.
    :param modifier: function which takes a tensor and returns a tensor of
    the same size.
    :param expected_result: array of same size as input_tensor.
    """

    # Some operations fail if the input is not a Variable - I'm not sure why.
    input_tensor = torch.autograd.Variable(torch.Tensor(input_data))
    output_tensor = torch.Tensor(expected_result)

    assert torch.equal(modifier(input_tensor).data, output_tensor)


def _noise_test():
    """Test whether noise is added properly. Due to the randomness involved,
    this test is not perfect. It checks that most values have changed,
    and that the largest change is less than or equal to the amount
    requested."""
    data = torch.autograd.Variable(torch.Tensor([0.0] * 100))
    result = noise_fn(1.0)(data).data

    positive = 0
    negative = 0

    for value in result:
        assert -1.0 <= value <= 1.0
        if value < 0.0:
            negative += 1
        elif value > 0.0:
            positive += 1

    # Ensure we're not left with zeroes, and that roughly the same number of
    # values have been increased as have been decreased.
    # These could fail if we're incredibly unlucky.
    assert positive + negative > 95
    assert 35 < positive < 65
    assert 35 < negative < 65


def _precision_test():
    """Test whether values are correctly rounded."""
    _check_equal([1.1, 0.6, 1.0, 1.5, 2.499], precision_fn(1.0),
                 [1.0, 1.0, 1.0, 2.0, 2.0])
    _check_equal([1.1, 0.6, 1.0, 1.5, 2.499], precision_fn(0.5),
                 [1.0, 0.5, 1.0, 1.5, 2.5])


def _cap_test():
    """Test whether upper bounds (on magnitudes) work correctly."""
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], cap_fn(5.0),
                 [-2.0, -1.0, 0.0, 1.0, 2.0])
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], cap_fn(1.0),
                 [-1.0, -1.0, 0.0, 1.0, 1.0])
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], cap_fn(0.5),
                 [-0.5, -0.5, 0.0, 0.5, 0.5])


def _threshold_test():
    """Test whether lower bounds (on magnitudes) work correctly."""
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], threshold_fn(0.5),
                 [-2.0, -1.0, 0.0, 1.0, 2.0])
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], threshold_fn(1.0),
                 [-2.0,  0.0, 0.0, 0.0, 2.0])
    _check_equal([-2.0, -1.0, 0.0, 1.0, 2.0], threshold_fn(5.0),
                 [ 0.0,  0.0, 0.0, 0.0, 0.0])


def _quantiser_test():
    _noise_test()
    _precision_test()
    _cap_test()
    _threshold_test()

    print("All tests passed.")


if __name__ == "__main__":
    _quantiser_test()
