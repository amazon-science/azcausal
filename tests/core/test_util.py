import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from azcausal.data import CaliforniaProp99
from azcausal.util import time_as_int


@pytest.fixture
def df():
    return CaliforniaProp99().df()


def test_time_to_idx(df):
    x = pd.Series([1990, 1980, 1970])
    assert_almost_equal(time_as_int(x), [2, 1, 0])
