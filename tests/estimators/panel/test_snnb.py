import pytest
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.snnb import SNNB


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


def test_snnb(panel):
    estimator = SNNB(random_state=RandomState(42))
    result = estimator.fit(panel)
    assert_almost_equal(result.effect.value, -25.522366305526276)



