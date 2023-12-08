import numpy as np
from numpy.testing import assert_almost_equal

from azcausal.core.error import Placebo, JackKnife, Bootstrap
from azcausal.data import CaliforniaProp99, Billboard
from azcausal.estimators.did import DID
from azcausal.estimators.event_study import EventStudy

california99 = CaliforniaProp99().panel()
billboard = Billboard().cdf()


def test_did_panel():
    from azcausal.estimators.panel.did import DID
    result = DID().fit(california99)
    assert_almost_equal(-27.349111083614947, result.effect.value)


def test_did_frame():
    from azcausal.estimators.did import DID
    cdf = CaliforniaProp99().cdf()
    result = DID().fit(cdf, se=True)
    assert_almost_equal(-27.349111083614947, result.effect.value)


def test_did_frame_bootstrap():
    from azcausal.estimators.did import DID
    cdf = CaliforniaProp99().cdf()
    estimator = DID()
    result = estimator.fit(cdf, se=True)

    estimator.error(result, Placebo(n_samples=11))
    estimator.error(result, JackKnife())
    estimator.error(result, Bootstrap(n_samples=11))

    assert_almost_equal(-27.349111083614947, result.effect.value)





def test_did_error_se_no_fail():
    from azcausal.estimators.panel.did import DID
    estimator = DID()
    result = estimator.fit(california99)

    estimator.error(result, Placebo(n_samples=11))
    estimator.error(result, JackKnife())
    estimator.error(result, Bootstrap(n_samples=11))


def test_did_billboard():
    from azcausal.estimators.did import DID
    estimator = DID()
    result = estimator.fit(billboard, se=True)
    assert np.allclose(result.effect.value, 6.5246)

    assert np.allclose(result.effect.se,  5.729, atol=1e-3)


def test_event_study():
    cdf = CaliforniaProp99().cdf()
    result_did = DID().fit(cdf, by_time=True)
    result_event = EventStudy(n_pre=0).fit(cdf)

    assert_almost_equal(result_did.effect.value, result_event.effect.value)

    # desired = result_did.effect.by_time.query("post == 1")["att"].values
    # actual = result_event.effect.by_time["att"].values
    #
    # assert_almost_equal(actual, desired)


def test_event_study_frame():
    cdf = CaliforniaProp99().cdf()
    result_event = EventStudy(n_pre=0).fit(cdf)
    assert_almost_equal(result_event["att"].value, -27.349111083614947)
