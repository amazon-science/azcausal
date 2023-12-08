from abc import abstractmethod
from os.path import dirname, realpath, join, exists
from urllib import request

import pandas as pd

from azcausal.core.frame import CausalDataFrame
from azcausal.core.panel import CausalPanel
from azcausal.util import time_as_int, to_panels


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

    def df(self):
        return pd.read_csv(self.path)

    @abstractmethod
    def remote(self):
        pass

    @abstractmethod
    def file(self):
        raise Exception("Each Dataset must ")


class CaliforniaProp99(DataSet):

    def raw(self):
        return pd.read_csv(self.path, delimiter=';')

    def remote(self):
        return "https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv"

    def file(self):
        return "california_prop99.csv"

    def df(self):
        return self.raw()

    @property
    def ctypes(self):
        return dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    def cdf(self):
        df = self.df()
        return CausalDataFrame(df).setup(**self.ctypes)

    def panel(self):
        ctypes = self.ctypes
        data = to_panels(self.df(), ctypes['time'], ctypes['unit'], [ctypes['outcome'], ctypes['intervention']])
        return CausalPanel(data).setup(**ctypes)


class Billboard(DataSet):

    def raw(self):
        return pd.read_csv(self.path)

    def remote(self):
        return "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/billboard_impact.csv"

    def file(self):
        return "billboard_impact.csv"

    def df(self):
        df = self.raw()
        df['month'] = df['jul'].map(lambda x: 'July' if x == 1 else 'May')
        df['time'] = time_as_int(df['month'])
        df['city'] = df['poa'].map(lambda x: 'Porto Alegre' if x == 1 else 'Florianopolis')

        return (df[['time', 'month', 'city', 'deposits']]
                .assign(post=lambda dx: (dx['month'] == 'July'))
                .assign(treatment=lambda dx: (dx['city'] == 'Porto Alegre'))
                .assign(intervention=lambda dx: dx['treatment'] & dx['post'])
                )

    def cdf(self):
        ctypes = dict(outcome='deposits', time='time', unit='city')
        cdf = CausalDataFrame(self.df()).setup(**ctypes)
        return cdf

    def panel(self):
        raise "Not available."
