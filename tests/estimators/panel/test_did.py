import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import Placebo, JackKnife, Bootstrap
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID, DIDRegressor, EventStudy


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()


@pytest.fixture
def df():
    return CaliforniaProp99().load()


@pytest.mark.parametrize("did", [DID(), DIDRegressor()])
def test_did(panel, did):
    result = did.fit(panel)
    assert_almost_equal(-27.349111083614947, result.effect.value)


@pytest.mark.parametrize("did", [DID(), DIDRegressor()])
def test_did_error_no_fail(panel, did):
    result = did.fit(panel)

    did.error(result, Placebo(n_samples=11))
    did.error(result, JackKnife())
    did.error(result, Bootstrap(n_samples=11))


def test_event_study(panel):
    result_did = DID().fit(panel)
    result_event = EventStudy(n_pre=0).fit(panel)

    assert_almost_equal(result_did.effect.value, result_event.effect.value)

    desired = result_did.effect.by_time.query("W == 1")["att"].values
    actual = result_event.effect.by_time["att"].values

    assert_almost_equal(actual, desired)


def test_event_study_frame(df):
    df = df.rename(columns=dict(Year="time", State="unit", PacksPerCapita="outcome", treated="intervention"))
    result_event = EventStudy(n_pre=0).fit(df)
    assert_almost_equal(result_event["att"].value, -27.349111083614947)
