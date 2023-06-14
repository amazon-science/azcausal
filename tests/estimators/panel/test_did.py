import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import Placebo, JackKnife, Bootstrap
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID, DIDRegressor, EventStudy

import numpy as np
@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


@pytest.mark.parametrize("did", [DID(), DIDRegressor()])
def test_did(panel, did):
    estm = did.fit(panel)
    assert_almost_equal(-27.349111083614947, estm["att"])


@pytest.mark.parametrize("did", [DID(), DIDRegressor()])
def test_did_error_no_fail(panel, did):
    estm = did.fit(panel)

    did.error(estm, Placebo(n_samples=11))
    did.error(estm, JackKnife())
    did.error(estm, Bootstrap(n_samples=11))


def test_event_study(panel):
    estm_did = DID().fit(panel)
    estm_event = EventStudy(n_pre=0).fit(panel)

    assert_almost_equal(estm_did["att"], estm_event["att"])
    assert_almost_equal(estm_did["data"]["att"].dropna().values, estm_event["data"]["att"].values)
