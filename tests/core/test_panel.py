import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from azcausal.util import to_matrix
from azcausal.data import CaliforniaProp99


@pytest.fixture
def data():
    return CaliforniaProp99()


def test_Y_shapes(data):
    pnl = data.panel()

    pnl.outcome.loc[2001] = 0.0
    pnl.outcome.loc[2002] = 0.0

    pnl.intervention.loc[2001] = 0
    pnl.intervention.loc[2001, "Wyoming"] = 1
    pnl.intervention.loc[2002] = 0

    time_pre = pnl.time(pre=True)
    assert_almost_equal(np.arange(1970, 1989), time_pre)

    time_post = pnl.time(post=True, trim=True)
    assert_almost_equal(np.arange(1989, 2002), time_post)

    time_post_no_trim = pnl.time(post=True)
    assert_almost_equal(np.arange(1989, 2003), time_post_no_trim)

    assert pnl.Y(pre=True).shape[1] == len(time_pre)
    assert pnl.Y(post=True, trim=True).shape[1] == len(time_post)


def test_panel_dates(data):
    pnl = data.panel()

    assert pnl.start == 1989
    assert pnl.earliest_start == 1989
    assert pnl.latest_start == 1989

    pnl.outcome.loc[2020] = 0.0
    pnl.intervention.loc[2020] = 0
    pnl.intervention.loc[2020, "Wyoming"] = 1

    assert pnl.latest_end == 2020
    assert pnl.earliest_end == 2000
    assert pnl.end is None


def test_replace_nan(data):
    df = data.load()

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita")
    assert outcome.isna().values.sum() == 10

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita", fillna=-1)
    assert outcome.isna().values.sum() == 0
    assert (outcome == -1).values.sum() == 10


def test_matrix_shapes(data):
    pnl = data.panel()

    print("sdfsf")
