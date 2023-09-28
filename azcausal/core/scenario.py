import json
from os import makedirs
from os.path import exists, join

import pandas as pd

from azcausal.core.estimator import results_from_outcome
from azcausal.core.panel import Panel
from azcausal.core.result import Result


class Scenario(object):

    def __init__(self,
                 true_outcome: pd.DataFrame,
                 treated_outcome: pd.DataFrame,
                 intervention: pd.DataFrame,
                 tags: dict = None
                 ) -> None:
        """


        Parameters
        ----------
        true_outcome
            The outcome without any effect (ground truth)

        treated_outcome
            The outcome where the intervention has taken place.

        intervention
            A dataframe indicating what time-unit cells have been treated (0: no intervention; 1: intervention)

        tags
            Additional keywords representing this scenario.

        """
        self.true_outcome = true_outcome
        self.treated_outcome = treated_outcome
        self.intervention = intervention
        self.tags = tags

    def result(self) -> Result:
        """
        The expected result for this scenario.
        """
        return results_from_outcome(self.treated_outcome, self.true_outcome, self.intervention)

    def panel(self) -> Panel:
        return Panel(self.treated_outcome, self.intervention)

    # store a scenario in a folder.
    def save(self, path: str) -> None:

        if not exists(path):
            makedirs(path, exist_ok=True)

        for name in ['true_outcome', 'treated_outcome', 'intervention']:
            obj = self.__dict__.get(name)
            if obj is not None:
                obj.to_csv(join(path, f"{name}.csv"))

        with open(join(path, f"tags.csv"), "w") as outfile:
            json.dump(self.tags, outfile)

    @classmethod
    # load a scenario from a folder
    def load(cls, path):
        args = []
        for name in ['true_outcome', 'treated_outcome', 'intervention']:
            obj = None

            file = join(path, f"{name}.csv")
            if exists(file):
                obj = pd.read_csv(file, index_col=0, parse_dates=True)

            args.append(obj)

        tags = None
        file = join(path, f"tags.csv")
        if exists(file):
            with open(join(path, f"tags.csv"), "r") as f:
                tags = json.loads(f.read())

        return cls(*args, tags=tags)


