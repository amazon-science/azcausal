from copy import deepcopy

import numpy as np
import pandas as pd

from azcausal.core.output import Output
from azcausal.core.summary import Summary
from azcausal.util import argmax


class Panel:

    def __init__(self,
                 outcome='outcome',
                 intervention='intervention',
                 confounders=None,
                 data=None,
                 mapping=None,
                 balanced=True) -> None:
        """
        The `Panel` object is a collection of data frames where the index is time and the columns are the units.

        Parameters
        ----------
        outcome : pd.DataFrame or str
            The outcome as `pd.DataFrame` or as string mapping to the data provided.

        intervention : pd.DataFrame or str
            The intervention as `pd.DataFrame` or as string mapping to the data provided.

        confounders : list
            A list of strings mapping to entries in the data.

        data : dict
            A dictionary where each key maps to a data frame.

        mapping : dict
            A mapping can be provided directly and overwrites the strings provided by outcome and intervention.

        balanced
            Strictly check the input data are balanced (do not contain any NaNs).

        """
        super().__init__()

        if data is None:
            data = dict()

        # a list of values to be considered as confounders.
        if confounders is None:
            confounders = list()

        # set the data directly if provided not via string
        if isinstance(intervention, pd.DataFrame):
            data['intervention'] = intervention
            intervention = 'intervention'
        if isinstance(outcome, pd.DataFrame):
            data['outcome'] = outcome
            outcome = 'outcome'

        # use the default mapping if not explicitly provided
        if mapping is None:
            mapping = dict(outcome=outcome, intervention=intervention, confounders=confounders)

        self.data = data
        self.mapping = mapping
        self.balanced = balanced

        self.check()

    def check(self):

        assert self.intervention is not None, "intervention needs to be provided"

        units = list(self.intervention.columns)
        time = list(self.intervention.index)

        for name, df in self.data.items():

            assert list(df.columns) == units, f"{name} need to have the same column names as the intervention data."
            assert list(df.index) == time, f"{name} need to have the same index as the intervention data."

            # check if the input is balanced
            if self.balanced:
                assert np.all(~np.isnan(df.values)), f"{name} contains `nan` values and thus is not balanced."

    def get_target(self, name):
        if name in self.mapping:
            name = self.mapping[name]
        return self.data.get(name)

    def set_target(self, name, df):
        if name in self.mapping:
            name = self.mapping[name]
        self.data[name] = df

        return self

    def with_targets(self, **kwargs):
        data = self.data.copy()
        for k, v in kwargs.items():
            data[k] = v

        return self.copy(deep=False, data=data)

    # select a specific value as outcome
    def select(self, value: str):

        # create a new mapping where value is outcome
        mapping = dict(self.mapping)
        mapping['outcome'] = value

        return Panel(data=self.data, mapping=mapping)

    # all values available
    @property
    def targets(self):
        return list(self.data.keys())

    def apply(self, f):
        data = {k: f(v) for k, v in self.data.items()}
        return self.copy(deep=False, data=data)

    def copy(self, deep=False, **kwargs):

        for k, v in self.__dict__.items():
            if k not in kwargs:
                kwargs[k] = v

        if deep:
            kwargs = deepcopy(kwargs)

        return Panel(**kwargs)

    # ----------------------------------------------- PANDAS -----------------------------------------------------

    # map the indicator function for the panel
    def __getitem__(self, key):
        return self.apply(lambda df: df[key])

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

    def to_frame(self,
                 index: bool = False,
                 labels: bool = True,
                 targets: list = None):
        """
        Converts all the panels back to a data frame which will always be balanced.

        Parameters
        ----------
        index
            If true, the result has the multi-index (time, unit).

        labels
            If true, then the original target names will be used.
            Otherwise, it will be a simple representation using (time, unit, outcome).

        targets
            What targets should be included in the panel

        Returns
        -------
        df
            A pd.DataFrame object created from the panel data.

        """

        # get an empty panel (balanced)
        df = pd.DataFrame({}, index=pd.MultiIndex.from_product([self.index, self.columns]))

        # if not provided use all targets
        if targets is None:
            targets = self.data.keys()

        # join each of the targets
        for target in targets:
            df = df.join(self.get_target(target).stack().to_frame(target), how="left")

        # remap the name of each target
        if not labels:
            df.index = df.index.set_names(["time", "unit"])
            df = df.rename(columns={self.mapping['outcome']: 'outcome', self.mapping['intervention']: 'intervention'})

        # reset the index if desired
        if not index:
            df = df.reset_index()

        return df

    # ----------------------------------------------- CORE -----------------------------------------------------

    def get(self, target, time=slice(None), units=slice(None), pre=False, post=False,
            treat=False, contr=False, to_numpy=False, transpose=False):
        """

        Parameters
        ----------
        target : str
            The value that is being returned of the panel, e.g. Y for observations or W for intervention values.
        time : slice or list
            Whether only a specific time range should be included.
        units : slice or list
            Whether only specific units shall be returned.
        pre : bool
            Whether the time should be set to the period before intervention
        post : bool
            Whether the time should be set after at least one intervention has been applied.
        treat : bool
            Whether only treatment units should be returned.
        contr : bool
            Whether only control units should be returned.
        to_numpy : bool
            Whether the results should be converted to a numpy array
        transpose : bool
            Whether the result should be transposed

        Returns
        -------

        """
        assert not (pre and post), "The usage of pre and post at the same time is not permitted."
        assert not (treat and contr), "The usage of treat and contr at the same time is not permitted."

        if pre:
            time = self.time(pre=True)
        elif post:
            time = self.time(post=True)

        if treat:
            units = self.w
        elif contr:
            units = ~self.w

        # query the data frame given the input
        dy = self.get_target(target)
        assert dy is not None, f"Target {target} not found. Targets available {self.targets}."

        # only get the subset that is requested
        dy = dy.loc[time, units]

        # convert to numpy if desired
        if to_numpy:
            dy = dy.values

        if transpose:
            dy = dy.T

        return dy

    # ----------------------------------------------- UNITS -----------------------------------------------------

    def units(self, treat=None, contr=None):
        """

        Parameters
        ----------
        treat : bool
            Whether only treatment units should be returned.
        contr : bool
            Whether only control units should be returned.

        Returns
        -------
        units : np.array
            The units of the panel (filters applied if desired)

        """
        units = np.array(self.intervention.columns)

        if (treat is True) or (contr is False):
            units = units[self.w]
        elif (treat is False) or (contr is True):
            units = units[~self.w]

        return units

    def n_units(self, **kwargs):
        return len(self.units(**kwargs))

    # ----------------------------------------------- TIME -----------------------------------------------------

    def trim(self):
        """
        Removes trailing observations with zero interventions (after the first intervention has already happened).

        Returns
        -------
        panel
            A new panel where trailing observations are removed.

        """
        return self.loc[:self.latest_end]

    def time(self, pre=False, post=False, contr=False, treat=False):
        """

        Parameters
        ----------
        pre : bool
            Time steps before the first treatment.
        post : bool
            Time steps after the first treatment.
        treat : bool
            Time steps where at least one treatment has occurred
        contr : bool
            Time steps where no treatment in any unit has occurred

        Returns
        -------
        time : np.array
            The time periods where the restrictions

        """
        time = np.array(self.intervention.index)

        if treat:
            time = time[self.wp]
        elif contr:
            time = time[~self.wp]
        elif pre:
            time = time[:self.wp.argmax()]
        elif post:
            time = time[self.wp.argmax():]

        return time

    def n_time(self, **kwargs):
        return len(self.time(**kwargs))

    @property
    def earliest_start(self):
        return self.time()[self.wp.argmax()]

    @property
    def latest_start(self):
        k = self.intervention.values.argmax(axis=0)[self.w].max()
        return self.time()[k]

    @property
    def start(self):
        if self.earliest_start == self.latest_start:
            return self.earliest_start

    @property
    def earliest_end(self):
        k = min([argmax(w, mode="last") for w in self.W()[self.w]])
        return self.time()[k]

    @property
    def latest_end(self):
        k = argmax(self.wp, mode="last")
        return self.time()[k]

    @property
    def end(self):
        if self.earliest_end == self.latest_end:
            return self.earliest_end

    # ----------------------------------------------- Values -----------------------------------------------------

    def Y(self, to_numpy=True, transpose=True, **kwargs):
        return self.get("outcome", to_numpy=to_numpy, transpose=transpose, **kwargs)

    def Y_as_block(self, **kwargs):
        return self.as_block("outcome", **kwargs)

    def W(self, to_numpy=True, transpose=True, **kwargs):
        return self.get("intervention", to_numpy=to_numpy, transpose=transpose, **kwargs)

    def W_as_block(self, **kwargs):
        return self.as_block("intervention", **kwargs)

    def as_block(self, value, to_numpy=True, transpose=True, **kwargs):
        kwargs["to_numpy"] = to_numpy
        kwargs["transpose"] = transpose

        pre_treat = self.get(value, pre=True, treat=True, **kwargs)
        post_treat = self.get(value, post=True, treat=True, **kwargs)
        pre_contr = self.get(value, pre=True, contr=True, **kwargs)
        post_contr = self.get(value, post=True, contr=True, **kwargs)

        return pre_treat, post_treat, pre_contr, post_contr

    @property
    def w(self):
        """
        Returns
        -------
        w : np.array
            A boolean array where `true` represents a unit is treated at some point in time, `false` otherwise.
        """
        return self.intervention.values.sum(axis=0) > 0

    @property
    def wp(self):
        """
        Returns
        -------
        wp : np.array
            A boolean array where `true` represents at least one unit is treated in the i-time step, `false` otherwise.
        """
        return self.intervention.values.sum(axis=1) > 0

    # ----------------------------------------------- CONVENIENCE -----------------------------------------------------

    @property
    def outcome(self):
        return self.get_target('outcome')

    @outcome.setter
    def outcome(self, df):
        return self.set_target('outcome', df)

    @property
    def intervention(self):
        return self.get_target('intervention')

    @intervention.setter
    def intervention(self, df):
        return self.set_target('intervention', df)

    def n_treatments(self):
        return len(np.unique(self.intervention.values)) - 1

    def n_interventions(self, treatment=None):
        if treatment is None:
            mask = self.intervention.values != 0
        else:
            mask = self.intervention.values == treatment
        return mask.sum()

    @property
    def n_pre(self):
        return self.n_time(pre=True)

    @property
    def n_post(self):
        return self.n_time(post=True)

    @property
    def n_treat(self):
        return self.n_units(treat=True)

    @property
    def n_contr(self):
        return self.n_units(contr=True)

    def counts(self):
        return (self.n_pre, self.n_post), (self.n_contr, self.n_treat)

    # ----------------------------------------------- OUTPUT -----------------------------------------------------

    def summary(self, **kwargs):
        output = (Output()
                  .text("Panel", align="center")
                  .texts([f"Time Periods: {self.n_time()} ({self.n_pre}/{self.n_post})", "total (pre/post)"])
                  .texts([f"Units: {self.n_units()} ({self.n_contr}/{self.n_treat})", "total (contr/treat)"])
                  )
        return Summary([output])
