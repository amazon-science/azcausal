from abc import abstractmethod
from os.path import dirname, realpath, join, exists
from urllib import request

import pandas as pd

from azcausal.core.panel import Panel
from azcausal.util import to_matrices


class DataSet:

    def __init__(self) -> None:
        super().__init__()
        self.download_if_not_exists()

    def local(self):
        dir = dirname(realpath(__file__))
        return join(dir, self.file())

    def download_if_not_exists(self, force=False):
        path = self.local()
        if force or not exists(path):
            request.urlretrieve(self.remote(), path)

    def load(self):
        return pd.read_csv(self.local())

    @abstractmethod
    def remote(self):
        pass

    @abstractmethod
    def file(self):
        pass


class CaliforniaProp99(DataSet):

    def load(self):
        return pd.read_csv(self.local(), delimiter=';')

    def panel(self):
        data = self.load()
        outcome, intervention = to_matrices(data, "Year", "State", "PacksPerCapita", "treated")
        return Panel(outcome, intervention)

    def remote(self):
        return "https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv"

    def file(self):
        return "california_prop99.csv"


class Abortion(DataSet):

    def load(self):
        return pd.read_stata(self.local())
    def panel(self):
        data = self.load()
        outcome, intervention = to_matrices(data, "Year", "State", "PacksPerCapita", "treated")
        return Panel(outcome, intervention)

    def remote(self):
        return "https://github.com/scunning1975/mixtape/raw/master/abortion.dta"

    def file(self):
        return "abortion.dta"
