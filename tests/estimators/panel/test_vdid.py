import pytest
from numpy.testing import assert_almost_equal

from azcausal.core.error import JackKnife
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.sdid import SDID
from azcausal.estimators.panel.vdid import vdid_panel


@pytest.fixture
def panel():
    return CaliforniaProp99().panel()

@pytest.fixture
def df():
    return (CaliforniaProp99()
            .df()
            .pipe(lambda dx: dx.assign(treatment=dx['State'].isin(dx.query('treated == 1')['State'].unique())))
            .pipe(lambda dx: dx.assign(post=dx['Year'].isin(dx.query('treated == 1')['Year'].unique())))
            .assign(total='total')
            )


def test_vdid(panel, df):
    estimator = DID()
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())
    assert_almost_equal(-27.349111083614947, result.effect.value)

    dte = vdid_panel(df, ['total'], 'PacksPerCapita', 'Year', 'State')
    te = dte['avg'].loc['total', 'PacksPerCapita']

    assert_almost_equal(te['te'], result.effect.value)
    assert_almost_equal(te['se'], result.effect.se)



def test_vdid_with_weights(panel, df):
    estimator = SDID()
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())

    weights = dict(treatment={False: result.effect['omega']},
                   post={False: result.effect['lambd']}
                   )

    dte = vdid_panel(df, ['total'], 'PacksPerCapita', 'Year', 'State', weights=weights)
    te = dte['avg'].loc['total', 'PacksPerCapita']

    assert_almost_equal(te['te'], result.effect.value)
    assert_almost_equal(te['se'], result.effect.se)


def test_vdid_zero_column(panel, df):
    panel.data['outcome']['VOID'] = 0.0
    panel.data['intervention']['VOID'] = 0

    panel.data['outcome']['VOID_T'] = 0.0
    panel.data['intervention']['VOID_T'] = panel.data['intervention']['California']

    estimator = DID()
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())

    treatment = df.groupby('treatment')['State'].unique()
    treatment[False] = list(treatment[False]) + ['VOID']
    treatment[True] = list(treatment[True]) + ['VOID_T']
    dims = dict(treatment=treatment)

    dte = vdid_panel(df, ['total'], 'PacksPerCapita', 'Year', 'State', dims=dims)
    te = dte['avg'].loc['total', 'PacksPerCapita']

    assert_almost_equal(te['te'], result.effect.value)
    assert_almost_equal(te['se'], result.effect.se)

