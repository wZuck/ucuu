from ucuu.decorator import ucuu


def print_without_ucuu():
    print("Print function running...")


print_without_ucuu = ucuu(
    "package_utils.print_ucuu_hello", ending_words="but now you get a ucuu"
)(print_without_ucuu)


def test_print_without_ucuu_but_with_ucuu():
    """
    Test case for the `print_without_ucuu` function.

    This test ensures that the `print_without_ucuu` function behaves as expected
    when invoked. The function's behavior should be validated in scenarios where
    it interacts with or without the presence of 'ucuu'.

    Note: Add specific assertions or mock setups if the function depends on external
    factors or has complex behavior.
    """
    print_without_ucuu()
