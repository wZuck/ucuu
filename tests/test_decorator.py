import pytest

from ucuu.decorator import ucuu


class TestPrint:
    """
    TestPrint contains test cases for the `ucuu` decorator applied to functions.

    Methods:
        test_func_no_endingwords:
            Test the `ucuu` decorator without providing `ending_words`.
            Verifies the decorator works with default behavior.

        test_func_with_endingwords:
            Test the `ucuu` decorator with `ending_words` set to "func_with_endingwords".
            Ensures the decorator handles custom ending words correctly.

        test_func_raise_errors:
            Test the `ucuu` decorator with `ending_words` set to "please raise errors".
            This test is expected to fail and raises an exception to validate error handling.
    """

    @ucuu("package_utils.print_ucuu_hello")
    def test_func_no_endingwords(self):
        print("Print function running...")

    @ucuu("package_utils.print_ucuu_hello", ending_words="func_with_endingwords")
    def test_func_with_endingwords(self):
        print("Print function running...")

    @pytest.mark.xfail(raises=Exception)
    @ucuu("package_utils.print_ucuu_hello", ending_words="please raise errors")
    def test_func_raise_errors(self):
        print("Print function running...")
