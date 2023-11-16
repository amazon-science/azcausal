from abc import abstractmethod
from os.path import dirname, realpath, join, exists
from urllib import request

import numpy as np
import pandas as pd

from azcausal.core.panel import Panel
from azcausal.util import to_matrices, treatment_and_post_from_intervention


class DataSet:

    def __init__(self, folder=None, cache=True) -> None:
        super().__init__()

        if folder is None:
            folder = join(dirname(realpath(__file__)), self.file())

        if cache:
            self.path = folder
            self.download_if_not_exists()
        else:
            self.path = self.remote()

    def download_if_not_exists(self, force=False):
        path = self.path
        if force or not exists(path):
            request.urlretrieve(self.remote(), path)

    def load(self):
        return pd.read_csv(self.path)

    @abstractmethod
    def remote(self):
        pass

    @abstractmethod
    def file(self):
        raise Exception("Each Dataset must ")


class CaliforniaProp99(DataSet):

    def load(self, rename=False):
        df = pd.read_csv(self.path, delimiter=';')
        df['post'] = (df['Year'] >= df.query("treated == 1")['Year'].min()).astype(int)
        df['treatment'] = (df['State'].isin(df.query("treated == 1")['State'].unique())).astype(int)

        if rename:
            df = df.rename(columns=dict(Year='time', State='unit', PacksPerCapita='outcome', treated='intervention'))

        return df

    def panel(self):
        data = self.load()
        data = to_matrices(data, "Year", "State", "PacksPerCapita", "treated")
        return Panel('PacksPerCapita', 'treated', data=data)

    def remote(self):
        return "https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv"

    def file(self):
        return "california_prop99.csv"


class Synthetic:
    def __init__(self, n_treat=10, n_contr=40, n_pre=90, n_post=10, att=0.5, noise=None):
        super().__init__()

        n_units = n_treat + n_contr
        n_time = n_pre + n_post

        units = np.array(["%02d" % k for k in range(1, n_units + 1)])
        time = np.arange(n_time) + 1900

        intervention = pd.DataFrame(index=time, columns=units).fillna(0).astype(int)
        intervention.iloc[-n_post:, :n_treat] = 1

        outcome = pd.DataFrame(index=time, columns=units).fillna(1.0)
        if noise is not None:
            outcome = outcome + np.random.normal(0, noise, size=(n_time, n_units))

        outcome.iloc[-n_post:, :n_treat] -= att

        self.outcome = outcome
        self.intervention = intervention

    def panel(self):
        return Panel(self.outcome, self.intervention)


class Billboard(DataSet):

    def load(self):
        df = pd.read_csv(self.path)
        df['outcome'] = df['deposits']
        df['time'] = df['jul'].map(lambda x: 'July' if x == 1 else 'May')
        df['unit'] = df['poa'].map(lambda x: 'Porto Alegre' if x == 1 else 'Florianopolis')
        return (df[['time', 'unit', 'outcome']]
                .assign(post=lambda dx: (dx['time'] == 'July'))
                .assign(treatment=lambda dx: (dx['unit'] == 'Porto Alegre').astype(int))
                .assign(intervention=lambda dx: dx['treatment'] & dx['post'])
                .astype(dict(post=int, treatment=int, intervention=int))
                )

    def panel(self):
        raise Exception("Not available for this data set.")

    def remote(self):
        return "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/billboard_impact.csv"

    def file(self):
        return "billboard_impact.csv"
