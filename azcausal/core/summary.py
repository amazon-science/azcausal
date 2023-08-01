from copy import deepcopy

from azcausal.core.output import Output


class Summary:

    def __init__(self,
                 sections,
                 title=None,
                 output=Output()) -> None:

        super().__init__()
        self.title = title
        self.sections = sections
        self.output = deepcopy(output)

    def header(self):
        self.output.tline()

    def footer(self):
        self.output.bline()

    def __str__(self):

        self.header()

        if self.title is not None:
            self.output.text(self.title, align="center")
            self.output.hrule()

        for i, section in enumerate(self.sections):
            self.output.append(section, inplace=True)

            if i < len(self.sections) - 1:
                self.output.hline()

        self.footer()

        return str(self.output)

