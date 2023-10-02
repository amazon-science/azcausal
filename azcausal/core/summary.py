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

        # if a title is provided add it as output to the very beginning
        if title is not None:
            output = Output()
            output.text(title, align="center")
            sections = [output] + sections

        self.sections = sections

    def __str__(self):

        # create a new output object
        output = Output()

        # print the top line
        output.tline()

        # for each section that is given
        for i, section in enumerate(self.sections):
            output.append(section, inplace=True)

            # always except for the last iteration
            if i < len(self.sections) - 1:
                output.hline()

        # print a footer section
        output.bline()

        return str(output)
