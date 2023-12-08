import pandas as pd
from numpy.random import RandomState

from azcausal.core.data import CausalData, make_balanced
from azcausal.core.panel import CausalPanel


class CausalDataFrame(pd.DataFrame, CausalData):
    _metadata = ["ctypes", "tags"]

    def __init__(self, data=None, **kwargs) -> None:
        super().__init__(data=data, **kwargs)

    @property
    def _constructor(self):
        return CausalDataFrame

    def setup(self,
              time: str = 'time',
              unit: str = 'unit',
              outcome: str = 'outcome',
              intervention: str = 'intervention',
              **kwargs):

        # create the causal data types to be used later
        ctypes = dict(outcome=outcome, time=time, unit=unit, intervention=intervention)

        # create a data frame only with the causal columns
        for ctype, column in ctypes.items():
            if column is not None and column in self.columns:
                self[ctype] = self[column]

        # if intervention not available just set all to zeros
        if 'intervention' not in self.columns:
            self['intervention'] = 0

        self.ctypes = ctypes
        self.reload()
        return self

    def filter(self, pre=None, post=None, contr=None, treat=None):
        df = self
        if (treat is True) or (contr is False):
            df = df.query(f"treatment == 1")
        if (treat is False) or (contr is True):
            df = df.query(f"treatment == 0")
        if (post is True) or (pre is False):
            df = df.query(f"post == 1")
        if (post is False) or (pre is True):
            df = df.query(f"post == 0")
        return df

    def reload(self, overwrite=False):
        assert 'intervention' in self.columns, "Needs the intervention column to load treatment and post."

        if overwrite or 'treatment' not in self.columns:
            treat_units = set(self[self.intervention != 0]['unit'].unique())
            self['treatment'] = self.unit.isin(treat_units)

        if overwrite or 'post' not in self.columns:
            start_post = self[self.intervention != 0]['time'].min()
            self['post'] = self['time'] >= start_post

        # convert all values to integers
        for column in ['intervention', 'post', 'treatment']:
            self[column] = self[column].astype(int)

        return self

    # ----------------------------------------------- UNITS -----------------------------------------------------

    def units(self, **kwargs):
        return list(self.filter(**kwargs)['unit'].unique())

    def by_unit(self, **kwargs):
        cdf = self.filter(**kwargs)
        return {k: v for k, v in cdf.groupby("unit")}

    # ----------------------------------------------- TIME -----------------------------------------------------

    def times(self, **kwargs):
        return list(self.filter(**kwargs)['time'].unique())

    def by_time(self, **kwargs):
        cdf = self.filter(**kwargs)
        return {k: v for k, v in cdf.groupby("time")}

    # ----------------------------------------------- CONVENIENCE -----------------------------------------------------

    def n_treatments(self):
        return len([e for e in self.intervention.unique() if e != 0])

    def n_interventions(self, treatment=None):
        if treatment is None:
            return (self.intervention != 0).sum()
        else:
            return (self.intervention == treatment).sum()

    def has_interventions(self, **kwargs):
        return self.n_interventions(**kwargs) > 0

    def select(self, column, ctype='outcome'):
        self.ctypes[ctype] = ctype
        return self.assign(**{ctype: lambda dx: dx[column]})

    # ----------------------------------------------- Pivot ----------------------------------------------------------

    def pivot(self, target='outcome', index='time', columns='unit', fillna=None, sort=True) -> pd.DataFrame:

        dy = self.set_index([index, columns])[target].unstack(columns, sort=False)

        if fillna is not None:
            dy.fillna(fillna, inplace=True)

        return dy

    # ----------------------------------------------- Misc -----------------------------------------------------------

    def length(self, **kwargs):
        return len(self.filter(**kwargs))

    def as_panel(self, target: str, **kwargs):
        return self.filter(**kwargs).pivot(target)

    def to_panel(self, targets=None, **kwargs):
        if targets is None:
            targets = [target for target in self.columns if target not in ['time', 'unit']]
        data = {target: self.as_panel(target, **kwargs) for target in targets}
        return CausalPanel(data=data).setup()

    def to_frame(self):
        return pd.DataFrame(self)

    def balance(self, columns=('time', 'unit'), sort=True):
        return make_balanced(self, columns=columns, sort=sort)

    def is_balanced(self):
        return self.n_units() * self.n_times() == len(self)

    def observed(self, treatment=1):
        return self.query(f"intervention == {treatment}")['outcome'].mean()

    # ----------------------------------------------- Sampling ---------------------------------------------------------

    def jackknife(self, units=None, seed=None, **kwargs):
        if units is None:
            units = self.units()

        if seed is None:
            seed = RandomState(seed).randint(low=0, high=len(units))

        unit = units[seed % len(units)]
        return self[self['unit'] != unit]

    def bootstrap(self, pool=None, seed=None, check=True, **kwargs):
        random_state = RandomState(seed)

        if pool is None:
            pool = self.by_unit()

        while True:
            sample = random_state.choice(list(pool.keys()), size=len(pool), replace=True)
            cdx = pd.concat([pool[s].assign(unit=i) for i, s in enumerate(sample)])

            if not check or (cdx.n_treat > 0 and cdx.n_contr > 0):
                return cdx

    def placebo(self, units=None, seed=None, **kwargs):
        random_state = RandomState(seed)

        # get the data by unit to sample from
        contr_units = self.by_unit(contr=True)
        treat_units = self.by_unit(treat=True)

        # create a random list of placebo units
        placebo = list(contr_units.keys())
        random_state.shuffle(placebo)

        dx = []
        for i, unit in enumerate(placebo):

            dfx = contr_units[unit]

            if len(treat_units) > 0:
                u = random_state.choice(list(treat_units.keys()))
                intervention = treat_units[u].intervention.values

                if len(intervention) == len(dfx):
                    dfx = dfx.assign(intervention=intervention, treatment=1)
                else:
                    raise "Not balanced data are not supported yet for Placebo tests."

                del treat_units[u]

            dx.append(dfx)

        cdx = pd.concat(dx).reload()

        return cdx

