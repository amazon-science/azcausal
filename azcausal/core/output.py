def wfill(s, n):
    return f"{' ' * n}{s}{' ' * n}"


class Output:

    def __init__(self, buffer=None) -> None:
        """
        An output object which allows to format the output and provides convenience for creating a table like printout.
        """

        super().__init__()
        self.width = 80

        if buffer is None:
            buffer = []

        self.buffer = buffer

    def write(self,
              line: str = ""):
        self.buffer.append(line)
        return line

    def append(self, output, inplace: bool = False):
        if inplace:
            return self.buffer.extend(output.buffer)
        else:
            return Output(buffer=self.buffer + output.buffer)

    def line(self,
             text: str = None,
             prefix: str = '|',
             suffix: str = '|',
             padding: int = 2,
             fill: str = " ",
             align: str = "center",
             margin: int = 0,
             write: bool
             = True):

        s = str(text) if text is not None else ""
        if len(s) > 0:
            s = wfill(s, padding)

        n = self.width - 2 - 2 * margin
        if align == "center":
            s = s.center(n, fill)
        elif align == "left":
            s = s.ljust(n, fill)
        elif align == "right":
            s = s.rjust(n, fill)
        else:
            raise Exception(f"Unknown align parameter: {align}")

        s = wfill(s, margin)
        s = f"{prefix}{s}{suffix}"

        if write:
            self.write(s)

        return s

    def tline(self, **kwargs):
        self.line(prefix='╭', suffix='╮', fill="─", **kwargs)
        return self

    def bline(self, **kwargs):
        self.line(prefix='╰', suffix='╯', fill="─", **kwargs)
        return self

    def hline(self, **kwargs):
        self.line(prefix='├', suffix='┤', fill="─", **kwargs)
        return self

    def whline(self, **kwargs):
        self.line(prefix='+', suffix='┤', fill="─", **kwargs)
        return self

    def hrule(self, **kwargs):
        self.line(prefix='├', suffix='┤', fill='=', **kwargs)
        return self

    def vspace(self):
        self.write()
        return self

    def text(self,
             text: str,
             align: str = "left",
             **kwargs):
        self.line(text=text, align=align, **kwargs)
        return self

    def texts(self, labels, aligns=None, **kwargs):
        raw = self.line(write=False, **kwargs)

        if len(labels) == 0:
            self.buffer.append(raw)
            return self

        if aligns is None:
            if len(labels) == 1:
                aligns = ["left"]
            elif len(labels) == 2:
                aligns = ["left", "right"]
            elif len(labels) == 3:
                aligns = ["left", "center", "right"]

        assert aligns is not None, "Alignments not available."
        assert len(aligns) == len(labels), "Alignments and labels must have the same size"

        s = [e for e in raw]

        for label, align in zip(labels, aligns):
            sp = self.line(text=label, align=align, write=False, **kwargs)
            if len(sp) != len(raw):
                self.line(text=label, align=align, write=False, **kwargs)
            assert len(sp) == len(raw)

            for k in range(len(raw)):
                if sp[k] != raw[k]:
                    s[k] = sp[k]

        s = "".join(s)
        self.write(s)

        return self

    def __str__(self):
        return "\n".join(self.buffer)
