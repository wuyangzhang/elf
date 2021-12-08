def data_provider(fn_data_provider):
    """
    Data provider decorator, allows another callable to provide the data for the test.
    Copy this helper function from https://pypi.org/project/unittest-data-provider/.
    """

    def test_decorator(fn):
        def repl(self, *args):
            for i in fn_data_provider():
                fn(self, *i)

        return repl

    return test_decorator