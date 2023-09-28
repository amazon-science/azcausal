import pytest

from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.snn import SNN


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


def test_snn(panel):
    estimator = SNN()
    result = estimator.fit(panel)
    assert (-30 < result.effect.value < -20)


