from azcausal.core.summary import Summary


class Result:

    def __init__(self, effects, data=None, estimator=None, title=None) -> None:
        super().__init__()

        self.effects = effects
        self.data = data

        self.estimator = estimator
        self.title = title

    @property
    def effect(self):
        if len(self.effects) > 0:
            return self.effects[next(iter(self.effects))]

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.effects:
            return self.effects[key]
        else:
            raise Exception(f"Key {key} not found.")

    def summary(self, title=None, cumulative=True, percentage=True, **kwargs):
        sections = []
        if hasattr(self.data, 'summary'):
            sections.append(self.data.summary(**kwargs))

        for effect in self.effects.values():

            sections.append(effect.summary(**kwargs))

            if percentage:
                perc = effect.percentage()
                sections.append(perc.summary(**kwargs))

            if cumulative and effect.multiplier is not None:
                cum = effect.cumulative(effect.multiplier)
                sections.append(cum.summary(**kwargs))

        return Summary(sections, title=title)

    @property
    def panel(self):
        return self.data

    # a setter function
    @panel.setter
    def panel(self, x):
        self.panel = x
