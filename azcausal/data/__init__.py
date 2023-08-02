from abc import abstractmethod
from os.path import dirname, realpath, join, exists
from urllib import request

import pandas as pd

from azcausal.core.panel import Panel
from azcausal.util import to_matrices


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

    def load(self):
        return pd.read_csv(self.path, delimiter=';')

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
