import numpy as np


class Format:

    def __init__(self) -> None:
        """
        The format how numbers should be represented.
        """
        super().__init__()

    def __call__(self,
                 x: float):
        return str(x)


class DefaultFormat(Format):

    def __call__(self, x):
        if x is None:
            s = ""
        elif isinstance(x, str):
            s = x
        else:
            v = np.abs(x)
            if v < 1:
                decimals = 6
            elif v < 10:
                decimals = 4
            elif v < 10000:
                decimals = 2
            else:
                decimals = 0

            f = '{' + f":,.{decimals}f" + '}'
            s = f.format(x)

        return s
