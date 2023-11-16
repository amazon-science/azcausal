import pytest
from numpy.random import RandomState

from azcausal.core.error import Bootstrap
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID

import numpy as np


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


def test_bootstrap_random(panel):
    result = DID().fit(panel)

    error = Bootstrap(n_samples=5)
    se, b = error.run(result)

    np.testing.assert_almost_equal(se['att'], 4.960760177933639)


def test_bootstrap_stratified(panel):
    result = DID().fit(panel)

    error = Bootstrap(mode='stratified', n_samples=5)
    se, b = error.run(result)

    np.testing.assert_almost_equal(se['att'], 1.9873233702850501)



