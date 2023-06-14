import pickle
from os.path import join, dirname, abspath

import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import JackKnife, Bootstrap, Placebo
from azcausal.core.panel import Panel
from azcausal.util import to_matrix
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID


@pytest.fixture
def data():
    return CaliforniaProp99()


@pytest.fixture
def sdid():
    return SDID()


def test_california_correct(sdid, data):
    pnl = data.panel()
    estm = sdid.fit(pnl)

    assert_almost_equal(77.3493584873376392, estm["solvers"]["lambd"]["f"])
    assert_almost_equal([0.3664706319364642, 0.2064530505601764, 0.4270763175033594],
                        estm["solvers"]["lambd"]["x"][-3:])
    assert estm["solvers"]["lambd"]["iter"] == 5

    assert_almost_equal(9.373915017, estm["solvers"]["omega"]["f"])
    assert_almost_equal([0.033569113191, 0.036667084791, 0.001386441800],
                        estm["solvers"]["omega"]["x"][-3:])
    assert estm["solvers"]["omega"]["iter"] == 10000

    assert_almost_equal(-15.6038278727338469, estm["att"])


def test_jackknife_correct(sdid, data):
    pnl = data.panel()
    # we need at least two treatment units for jackknife
    pnl.intervention["Wyoming"].loc[1989:] = 1

    estm = sdid.fit(pnl)
    assert_almost_equal(-4.1538623012790161, estm["att"])

    error = sdid.error(estm, JackKnife())
    assert_almost_equal(17.867241960405, error["se"])


def test_placebo_no_fail(sdid, data):
    pnl = data.panel()
    estm = sdid.fit(pnl)
    sdid.error(estm, Placebo(n_samples=2))


def test_bootstrap_no_fail(sdid, data):
    pnl = data.panel()
    estm = sdid.fit(pnl)
    sdid.error(estm, Bootstrap(n_samples=2))


def add_to_treatment(df, units, start):
    def f(x):
        if x["Year"] >= start and x["State"] in units:
            x["treated"] = 1
        return x

    return df.apply(f, axis=1)


def from_r():
    with open(join(dirname(abspath(__file__)), 'sdid', 'correct', 'synthdid.pkl'), 'rb') as f:
        return pickle.load(f)


def data_generator():
    df = CaliforniaProp99().load()
    units = df["State"].unique()

    for k in range(-1, 10):
        if k >= 0:
            yield add_to_treatment(df, units[:k], 1989)


@pytest.mark.parametrize("df,correct", zip(data_generator(), from_r()))
def test_correctness_comprehensive(sdid, df, correct):
    outcome = to_matrix(df, "Year", "State", "PacksPerCapita")
    intervention = to_matrix(df, "Year", "State", "treated")
    pnl = Panel(outcome, intervention)

    estm = sdid.fit(pnl)
    assert_almost_equal(correct["tau_hat"], estm["att"])

    assert_almost_equal(correct["lambd"], estm["lambd"])
    assert_almost_equal(correct["f_min_lambd"], estm["solvers"]["lambd"]["f"].min())

    assert_almost_equal(correct["omega"], estm["omega"])
    assert_almost_equal(correct["f_min_omega"], estm["solvers"]["omega"]["f"].min())

    error = sdid.error(estm, JackKnife())

    if pnl.n_treat > 1:
        assert_almost_equal(correct["se"], error["se"])


# @pytest.mark.parametrize("method", [Placebo(n_samples=5000),
#                                     Bootstrap(n_samples=5000)])
# def test_placebo_dist(sdid, data, method):
#     from azcausal.core.parallelize import Joblib
#
#     with open(join(dirname(abspath(__file__)), 'sdid', 'correct' f'{method.__class__.__name__.lower()}.pkl'), 'rb') as f:
#         correct = pickle.load(f)
#
#     pnl = data.panel()
#     pnl.treatment["Wyoming"].loc[1989:] = 1
#
#     estm = sdid.fit(pnl)
#     error = sdid.error(estm, method, parallelize=Joblib())
#
#     import matplotlib.pyplot as plt
#     plt.hist(correct, bins=101, density=True)
#     plt.hist(error["estms"], bins=101, density=True)
#     plt.show()
