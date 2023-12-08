import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.data import CausalData
from azcausal.util import zeros_like


class CausalPanel(CausalData):

    def __init__(self, data=None) -> None:
        """
        The causal panel object providing additional information to a data frame.

        Parameters
        ----------

        data : dict
            A dictionary where each key maps to a data frame.


        """
        super().__init__()
        if data is None:
            data = dict()
        self.data = data

    def setup(self,
              outcome: str = 'outcome',
              intervention: str = 'intervention',
              **kwargs):

        # use the default mapping if not explicitly provided
        self.ctypes = dict(outcome=outcome, intervention=intervention)

        self.assign(inplace=True, **{k: self[v] for k, v in self.ctypes.items()})
        assert self.intervention is not None, "intervention needs to be provided"

        return self

    def filter(self, pre=None, post=None, contr=None, treat=None, target=None, as_dict=False):

        w = np.full(len(self.columns), True)
        if (treat is True) or (contr is False):
            w &= self.treat
        if (treat is False) or (contr is True):
            w &= (~self.treat)

        wp = np.full(len(self.index), True)
        if (post is True) or (pre is False):
            wp &= self.post
        if (post is False) or (pre is True):
            wp &= (~self.post)

        if target is not None:
            return self.data[target].loc[wp, w]
        else:
            return self.apply(lambda dy: dy.loc[wp, w], as_dict=as_dict)

    @property
    def targets(self):
        return list(self.data.keys())

    def __getitem__(self, key):
        return self.data.get(key)

    def assign(self, inplace=False, **kwargs):

        def f(data):
            for k, v in kwargs.items():

                # the user can also provide a lambda function
                if callable(v):
                    v = v(self)

                # check whether the assignment is correct
                if 'intervention' in self.data:
                    assert np.all(
                        v.columns == self.intervention.columns), f"{k} has different columns than intervention"
                    assert np.all(v.index == self.intervention.index), f"{k} has a different index than intervention"

                # copy over to the data object
                data[k] = v

        if inplace:
            f(self.data)
            return self
        else:
            data = dict(self.data)
            f(data)
            return CausalPanel(data=data)

    def apply(self, func, targets=None, as_dict=False):
        if targets is None:
            targets = self.data.keys()

        data = {target: func(self.data[target]) for target in targets}

        if as_dict:
            return data
        else:
            return CausalPanel(data=data)

    # ----------------------------------------------- UNITS -----------------------------------------------------

    def units(self, **kwargs):
        return list(self.filter(**kwargs, target='intervention').columns)

    def times(self, **kwargs):
        return list(self.filter(**kwargs, target='intervention').index)

    @property
    def treat(self):
        """
        Returns
        -------
        w : np.array
            A boolean array where `true` represents a unit is treated at some point in time, `false` otherwise.
        """
        return self.intervention.values.sum(axis=0) > 0

    @property
    def post(self):
        """
        Returns
        -------
        wp : np.array
            A boolean array where `true` represents at least one unit is treated in the i-time step, `false` otherwise.
        """
        return self.intervention.values.sum(axis=1) > 0

    # ----------------------------------------------- CONVENIENCE -----------------------------------------------------

    @property
    def outcome(self) -> pd.DataFrame:
        return self['outcome']

    @outcome.setter
    def outcome(self, df):
        self.data['outcome'] = df

    @property
    def intervention(self) -> pd.DataFrame:
        return self['intervention']

    @intervention.setter
    def intervention(self, df):
        self.data['intervention'] = df

    def n_treatments(self):
        return len(np.unique(self.intervention.values)) - 1

    def n_interventions(self, treatment=None):
        if treatment is None:
            mask = self.intervention.values != 0
        else:
            mask = self.intervention.values == treatment
        return mask.sum()

    # ----------------------------------------------- PANDAS -----------------------------------------------------

    @property
    def columns(self):
        return self.intervention.columns

    @property
    def shape(self):
        return self.intervention.shape

    @property
    def index(self):
        return self.intervention.index

    @property
    def loc(self):
        panel = self

        class Index:
            def __getitem__(self, key):
                return panel.apply(lambda df: df.loc[key])

        return Index()

    @property
    def iloc(self):
        panel = self

        class Index:
            def __getitem__(self, key):
                return panel.apply(lambda df: df.iloc[key])

        return Index()

    # ----------------------------------------------- CONVERT -------------------------------------------------------
    def to_frame(self,
                 index: bool = False,
                 labels: bool = True,
                 targets: list = None):

        # get an empty panel (balanced)
        df = pd.DataFrame({}, index=pd.MultiIndex.from_product([self.index, self.columns]))

        # if not provided use all targets
        if targets is None:
            targets = self.data.keys()

        # join each of the targets
        for target in targets:
            df = df.join(self.data[target].stack().to_frame(target), how="left")

        # remap the name of each target
        if not labels:
            df.index = df.index.set_names(["time", "unit"])
            df = df.rename(columns={self.ctypes['outcome']: 'outcome', self.ctypes['intervention']: 'intervention'})

        # set the columns for treatment and post (can be useful for some analysis)
        df = (df
              .assign(post=lambda dx: df.index.get_level_values(0).isin(self.times(post=True)))
              .assign(treatment=lambda dx: df.index.get_level_values(1).isin(self.units(treat=True)))
              .astype(dict(post=int, treatment=int))
              )

        # reset the index if desired
        if not index:
            df = df.reset_index()

        return df

    def observed(self, treatment=1):
        return ((self.intervention == treatment).values * self.outcome.values).sum() / self.n_interventions(
            treatment=treatment)

    def jackknife(self, seed=None, **kwargs):
        n = len(self.columns)
        if seed is None:
            seed = RandomState().choice(np.arange(n))
        return self.iloc[:, np.delete(np.arange(n), seed % n)]

    def bootstrap(self, seed=None, n_units=None, replace=True, check=True, **kwargs):
        if check:
            assert self.n_treat > 0 and self.n_contr > 0, "At least one control and one treatment unit is needed."

        if n_units is None:
            n_units = len(self.columns)

        random_state = RandomState(seed)

        while True:
            x = random_state.choice(np.arange(len(self.columns)), size=n_units, replace=replace)
            panel = self.iloc[:, np.sort(x)]

            if not check or (panel.n_treat > 0 and panel.n_contr > 0):
                return panel

    def placebo(self, seed=None, n_treat=None, **kwargs):
        if n_treat is None:
            n_treat = self.n_treat
        assert n_treat > 0, "The number of placebo units needs to be larger than 0."

        panel = self.filter(contr=True)

        n = panel.n_units()
        random_state = RandomState(seed)

        # randomly draw placebo units from control
        placebo = random_state.choice(np.arange(n), size=n_treat, replace=False)

        # now create the intervention which simply randomizes the treatment given outcome
        intervention = zeros_like(panel.outcome)
        intervention.iloc[:, placebo] = self.filter(target="intervention", treat=True).values

        return panel.assign(intervention=intervention)


class Panel(CausalPanel):
    pass


