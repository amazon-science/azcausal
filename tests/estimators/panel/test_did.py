import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import Placebo, JackKnife, Bootstrap
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


def test_did(panel):
    did = DID()
    estm = did.fit(panel)

    assert_almost_equal(-27.349111083614947, estm["att"])


def test_did_error_no_fail(panel):
    did = DID()
    estm = did.fit(panel)

    did.error(estm, Placebo())
    did.error(estm, JackKnife())
    did.error(estm, Bootstrap())
