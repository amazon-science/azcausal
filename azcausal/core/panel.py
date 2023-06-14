import numpy as np
import pandas as pd
from azcausal.util import argmax


class Panel:

    def __init__(self, outcome, intervention) -> None:
        """
        A collection of different panel observations.

        Parameters
        ----------
        outcome : pd.DataFrame
            The observations
        intervention : pd.DataFrame
            The treatment values, where `true` indicates treatment and `false` does not.
        """
        super().__init__()
        self.outcome = outcome
        self.intervention = intervention

        self.check()

    def __getitem__(self, key):
        return Panel(self.outcome[key], self.intervention[key])

    @property
    def loc(self):
        outcome, intervention = self.outcome, self.intervention

        class Index:
            def __getitem__(self, key):
                return Panel(outcome.loc[key], intervention.loc[key])

        return Index()

    @property
    def iloc(self):
        outcome, intervention = self.outcome, self.intervention

        class Index:
            def __getitem__(self, key):
                return Panel(outcome.iloc[key], intervention.iloc[key])

        return Index()

    def check(self):

        # check if they have the same shape
        assert self.outcome.shape == self.intervention.shape, "The shape of outcome and intervention need to be identical."
        assert np.all(
            self.outcome.columns == self.intervention.columns), "The columns of `outcome` and `intervention` must be identical"

        # check for `nan` values
        assert not np.any(np.isnan(self.Y())), "The outcome data set contains `nan` values."
        assert not np.any(np.isnan(self.W())), "The intervention data set contains `nan` values."

    def to_frame(self, index=True, treatment=False, rtime=True):
        dy = self.outcome.unstack().to_frame("outcome")
        dy = dy.join(self.intervention.unstack().to_frame("intervention"))
        dy.index = dy.index.set_names(["unit", "time"])
        dy = dy.reset_index()

        if treatment:
            label = treatment if isinstance(treatment, str) else "treatment"
            dy[label] = dy["unit"].isin(set(self.units(treat=True))).astype(int)

        if rtime:
            time = self.time()
            label = rtime if isinstance(rtime, str) else "rtime"
            time_to_index = pd.DataFrame({label: np.arange(len(time))}, index=pd.Index(time, name="time"))
            dy = dy.merge(time_to_index, on="time")

        if index:
            dy = dy.set_index(["unit", "time"])

        return dy

    def get(self, value, time=slice(None), units=slice(None), pre=False, post=False, trim=False,
            treat=False, contr=False, to_numpy=False, transpose=False):
        """

        Parameters
        ----------
        value : str
            The value that is being returned of the panel, e.g. Y for observations or W for intervention values.
        time : slice or list
            Whether only a specific time range should be included.
        units : slice or list
            Whether only specific units shall be returned.
        pre : bool
            Whether the time should be set to the period before intervention
        post : bool
            Whether the time should be set after at least one intervention has been applied.
        trim : bool
            Removes trailing time periods where no interventions in any units have occurred.
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
            time = self.time(pre=True, trim=trim)
        elif post:
            time = self.time(post=True, trim=trim)
        elif trim:
            time = self.time(trim=trim)

        if treat:
            units = self.w
        elif contr:
            units = ~self.w

        # query the data frame given the input
        dy = self.__dict__[value].loc[time, units]

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
        units = np.array(self.outcome.columns)

        if (treat is True) or (contr is False):
            units = units[self.w]
        elif (treat is False) or (contr is True):
            units = units[~self.w]

        return units

    def n_units(self, **kwargs):
        return len(self.units(**kwargs))

    # ----------------------------------------------- TIME -----------------------------------------------------

    def time(self, pre=False, post=False, contr=False, treat=False, trim=False):
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
        trim : bool
            Removes trailing time periods where no treatments in any units have occurred.

        Returns
        -------
        time : np.array
            The time periods where the restrictions

        """
        time = np.array(self.outcome.index)

        if trim:
            t = np.array(self.wp)
            t[:t.argmax()] = True
            t = ~t
        else:
            t = np.full(len(time), False)

        if treat:
            time = time[self.wp & ~t]
        elif contr:
            time = time[~self.wp & ~t]
        elif pre:
            time = time[:self.wp.argmax()]
        elif post:
            if trim and t.sum() > 0:
                time = time[self.wp.argmax():t.argmax()]
            else:
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

    def copy(self):
        return Panel(self.outcome.copy(), self.intervention.copy())

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
