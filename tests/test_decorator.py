import pytest

from ucuu.decorator import ucuu

class TestPrintInClass:
    @ucuu("package_utils.print_ucuu_hello")
    def test_func_no_endingwords(self):
        print("PrintInClass function running...")

    @ucuu("package_utils.print_ucuu_hello", ending_words = "func_with_endingwords")
    def test_func_with_endingwords(self):
        print("PrintInClass function running...")

    @ucuu("package_utils.print_ucuu_hello", ending_words = "please raise errors")
    def test_func_raise_errors(self):
        print("PrintInClass function running...")

