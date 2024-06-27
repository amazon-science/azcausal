class SolverException(Exception):

    def __init__(self, *args, exceptions=None, **kwargs):
        super().__init__(*args)
        self.exceptions = exceptions


