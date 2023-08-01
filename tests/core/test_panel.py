import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from azcausal.util import to_matrix
from azcausal.data import CaliforniaProp99


@pytest.fixture
def data():
    return CaliforniaProp99()


def test_Y_shapes(data):
    panel = data.panel()

    panel.outcome.loc[2001] = 0.0
    panel.outcome.loc[2002] = 0.0

    panel.intervention.loc[2001] = 0
    panel.intervention.loc[2001, "Wyoming"] = 1
    panel.intervention.loc[2002] = 0

    time_pre = panel.time(pre=True)
    assert_almost_equal(np.arange(1970, 1989), time_pre)

    time_post = panel.time(post=True, trim=True)
    assert_almost_equal(np.arange(1989, 2002), time_post)

    time_post_no_trim = panel.time(post=True)
    assert_almost_equal(np.arange(1989, 2003), time_post_no_trim)

    assert panel.Y(pre=True).shape[1] == len(time_pre)
    assert panel.Y(post=True, trim=True).shape[1] == len(time_post)


def test_panel_dates(data):
    panel = data.panel()

    assert panel.start == 1989
    assert panel.earliest_start == 1989
    assert panel.latest_start == 1989

    panel.outcome.loc[2020] = 0.0
    panel.intervention.loc[2020] = 0
    panel.intervention.loc[2020, "Wyoming"] = 1

    assert panel.latest_end == 2020
    assert panel.earliest_end == 2000
    assert panel.end is None


def test_replace_nan(data):
    df = data.load()

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita")
    assert outcome.isna().values.sum() == 10

    outcome = to_matrix(df.iloc[:-10], "Year", "State", "PacksPerCapita", fillna=-1)
    assert outcome.isna().values.sum() == 0
    assert (outcome == -1).values.sum() == 10


