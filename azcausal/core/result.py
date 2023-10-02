from azcausal.core.panel import Panel
from azcausal.core.summary import Summary


class Result:

    def __init__(self,
                 effects: dict,
                 data: dict = None,
                 panel: Panel = None,
                 estimator=None) -> None:
        """
        A result object returned by an estimator.

        Parameters
        ----------
        effects
            A dictionary of effects measured by the estimator.
        data
            A dictionary with additional data (across effects)
        panel
            The panel based on which the estimates are based on.
        estimator
            The estimator that has been used.
        """
        super().__init__()

        self.effects = effects
        self.data = data
        self.panel = panel

        self.estimator = estimator

    @property
    def effect(self):
        # this always return the first effect of the dictionary -- can be used as a shortcut
        if len(self.effects) > 0:
            return self.effects[next(iter(self.effects))]

    # direct access to effects using the indicator notation
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.effects:
            return self.effects[key]
        else:
            raise Exception(f"Key {key} not found.")

    def summary(self,
                title: str = None,
                cumulative: bool = True,
                percentage: bool = True,
                **kwargs):
        """
        A `Summary` object for this result. This provides a convenient printout once the result has been returned
        by an estimator.

        Parameters
        ----------
        title
            The title of the summary.
        cumulative
            Whether the cumulative result should be shown.
        percentage
            Whether the percentage result should be shown.
        kwargs
            Additional keywords provided to the summary methods, e.g. dict(conf=90) for a difference confidence level.

        Returns
        -------
        summary
            A summary object.

        """
        sections = []
        if hasattr(self.panel, 'summary'):
            sections.extend(self.panel.summary(**kwargs).sections)

        for effect in self.effects.values():

            sections.append(effect.summary(**kwargs))

            if percentage:
                perc = effect.percentage()
                sections.append(perc.summary(**kwargs))

            if cumulative and effect.multiplier is not None:
                cum = effect.cumulative(effect.multiplier)
                sections.append(cum.summary(**kwargs))

        return Summary(sections, title=title)
