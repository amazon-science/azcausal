from azcausal.core.output import Output


class Summary:

    def __init__(self,
                 sections: list,
                 title: str = None) -> None:
        """
        A summary object can be used to format multiple sections into one output and conveniently adding
        header, footer, as well as separation lines.

        Parameters
        ----------
        sections
            A list of sections that should be printed.

        title
            What title should be printed in the header.

        """

        super().__init__()
        self.title = title
        self.sections = sections

    def __str__(self):

        # create a new output object
        output = Output()

        # print the top line
        output.tline()

        # print a header section if a title is provided
        if self.title is not None:
            output.text(self.title, align="center")
            output.hrule()

        # for each section that is given
        for i, section in enumerate(self.sections):
            output.append(section, inplace=True)

            # always except for the last iteration
            if i < len(self.sections) - 1:
                output.hline()

        # print a footer section
        output.bline()

        return str(output)
