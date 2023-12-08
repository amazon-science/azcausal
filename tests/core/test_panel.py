import numpy as np
import pytest

from azcausal.data import CaliforniaProp99
from azcausal.util import to_matrix


@pytest.fixture
def data():
    return CaliforniaProp99()


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


def test_to_frame(data, panel):
    dx = data.df().set_index(["Year", "State"]).sort_index()
    df = panel.to_frame(index=True).sort_index()
    assert np.all(df['PacksPerCapita'] == dx['PacksPerCapita'])


def test_replace_nan(data):
    df = data.df()

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita")
    assert outcome.isna().values.sum() == 10

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita", fillna=-1)
    assert outcome.isna().values.sum() == 0
    assert (outcome == -1).values.sum() == 10
