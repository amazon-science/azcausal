from abc import abstractmethod
from copy import deepcopy, copy

import pandas as pd

from azcausal.core.output import Output
from azcausal.core.summary import Summary


class CausalData:

    def __init__(self) -> None:
        super().__init__()
        self.ctypes = None
        self.tags = dict()

    @abstractmethod
    def setup(self,
              time: str = 'time',
              unit: str = 'unit',
              outcome: str = 'outcome',
              intervention: str = 'intervention',
              **kwargs):
        pass

    def select(self, value: str):
        ctypes = dict(self.ctypes)
        ctypes['outcome'] = value
        return self.copy().setup(**ctypes)

    @abstractmethod
    def filter(self, pre=None, post=None, contr=None, treat=None, **kwargs):
        pass

    @abstractmethod
    def units(self, **kwargs):
        pass

    @abstractmethod
    def times(self, **kwargs):
        pass

    @abstractmethod
    def n_interventions(self, **kwargs):
        pass

    @abstractmethod
    def n_treatments(self, **kwargs):
        pass

    def n_units(self, **kwargs):
        return len(self.units(**kwargs))

    def n_times(self, **kwargs):
        return len(self.times(**kwargs))

    @property
    def n_pre(self):
        return self.n_times(pre=True)

    @property
    def n_post(self):
        return self.n_times(post=True)

    @property
    def n_treat(self):
        return self.n_units(treat=True)

    @property
    def n_contr(self):
        return self.n_units(contr=True)

    def counts(self):
        return (self.n_pre, self.n_post), (self.n_contr, self.n_treat)

    def summary(self, **kwargs):
        output = (Output()
                  .text("Panel", align="center")
                  .texts([f"Time Periods: {self.n_times()} ({self.n_pre}/{self.n_post})", "total (pre/post)"])
                  .texts([f"Units: {self.n_units()} ({self.n_contr}/{self.n_treat})", "total (contr/treat)"])
                  )
        return Summary([output])

    def copy(self, deep=False):
        return deepcopy(self) if deep else copy(self)

    def jackknife(self, seed=None, **kwargs):
        pass

    def bootstrap(self, seed=None, **kwargs):
        pass

    def placebo(self, seed=None, **kwargs):
        pass


def make_balanced(df: pd.DataFrame,
                  columns: list,
                  sort: bool = True):
    """
    Making a data frame balanced by ensuring observations for all time periods.

    Parameters
    ----------
    df
        The data frame object.
    columns
        The columns to use for balancing.
    sort
        Whether the data frame should also be sorted by the columns.

    Returns
    -------
    df
        A balanced data frame.

    """
    x = [list(df[c].drop_duplicates()) for c in columns]
    if sort:
        x = [sorted(c) for c in x]
    index = pd.MultiIndex.from_product(x, names=columns)
    return df.set_index(list(columns)).reindex(index).reset_index()
