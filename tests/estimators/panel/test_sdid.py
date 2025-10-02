import pickle
from os.path import join, dirname, abspath

import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import JackKnife, Bootstrap, Placebo
from azcausal.core.panel import CausalPanel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID
from azcausal.util import to_matrices


@pytest.fixture
def data():
    return CaliforniaProp99()


@pytest.fixture
def sdid():
    return SDID()


def test_california_correct(sdid, data):
    panel = data.panel()

    result = sdid.fit(panel)
    effect = result.effect

    assert_almost_equal(effect["solvers"]["lambd"]["f"], 77.3493584873376392)
    assert_almost_equal(effect["solvers"]["lambd"]["x"][-3:], [0.3664706319364642, 0.2064530505601764, 0.4270763175033594])
    assert effect["solvers"]["lambd"]["iter"] == 5

    assert_almost_equal(effect["solvers"]["omega"]["f"], 9.373915017)
    assert_almost_equal(effect["solvers"]["omega"]["x"][-3:], [0.033569113191, 0.036667084791, 0.001386441800])
    assert effect["solvers"]["omega"]["iter"] == 10000

    assert_almost_equal(effect.value, -15.6038278727338469)


def test_jackknife_correct(sdid, data):
    panel = data.panel()
    # we need at least two treatment units for jackknife
    panel.intervention["Wyoming"].loc[1989:] = 1

    result = sdid.fit(panel)
    assert_almost_equal(-4.1538623012790161, result.effect.value)

    sdid.error(result, JackKnife())
    assert_almost_equal(17.867241960405, result.effect.se)


def test_placebo_no_fail(sdid, data):
    panel = data.panel()
    estm = sdid.fit(panel)
    sdid.error(estm, Placebo(n_samples=2))


def test_bootstrap_no_fail(sdid, data):
    panel = data.panel()
    estm = sdid.fit(panel)
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
    df = CaliforniaProp99().df()
    units = df["State"].unique()

    for k in range(-1, 10):
        if k >= 0:
            yield add_to_treatment(df, units[:k], 1989)


@pytest.mark.parametrize("df,correct", zip(data_generator(), from_r()))
def test_correctness_comprehensive(sdid, df, correct):
    data = to_matrices(df, "Year", "State", "PacksPerCapita", "treated")
    panel = CausalPanel(data=data).setup(outcome='PacksPerCapita', intervention='treated')

    result = sdid.fit(panel)
    effect = result.effect

    assert_almost_equal(correct["tau_hat"], effect.value)

    assert_almost_equal(correct["lambd"], effect["lambd"])
    assert_almost_equal(correct["f_min_lambd"], effect["solvers"]["lambd"]["f"].min())

    assert_almost_equal(correct["omega"], effect["omega"])
    assert_almost_equal(correct["f_min_omega"], effect["solvers"]["omega"]["f"].min())

    sdid.error(result, JackKnife())

    if panel.n_treat > 1:
        assert_almost_equal(correct["se"], result.effect.se)


# @pytest.mark.parametrize("method", [Placebo(n_samples=5000),
#                                     Bootstrap(n_samples=5000)])
# def test_placebo_dist(sdid, data, method):
#     from azcausal.core.parallelize import Joblib
#
#     with open(join(dirname(abspath(__file__)), 'sdid', 'correct' f'{method.__class__.__name__.lower()}.pkl'), 'rb') as f:
#         correct = pickle.load(f)
#
#     panel = data.panel()
#     panel.treatment["Wyoming"].loc[1989:] = 1
#
#     estm = sdid.fit(panel)
#     error = sdid.error(estm, method, parallelize=Joblib())
#
#     import matplotlib.pyplot as plt
#     plt.hist(correct, bins=101, density=True)
#     plt.hist(error["estms"], bins=101, density=True)
#     plt.show()
