from numpy.testing import assert_almost_equal

from azcausal.core.error import JackKnife
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.vdid import vdid_panel

california99 = CaliforniaProp99().panel()


def test_vdid():
    estimator = DID()
    result = estimator.fit(california99)
    estimator.error(result, JackKnife())
    assert_almost_equal(-27.349111083614947, result.effect.value)

    df = (CaliforniaProp99()
          .df()
          .pipe(lambda dx: dx.assign(treatment=dx['State'].isin(dx.query('treated == 1')['State'].unique())))
          .pipe(lambda dx: dx.assign(post=dx['Year'].isin(dx.query('treated == 1')['Year'].unique())))
          .assign(total='total')
          )

    dte = vdid_panel(df, ['total'], 'PacksPerCapita', 'Year', 'State')
    te = dte['cum'].loc['total', 'PacksPerCapita']

    assert_almost_equal(te['te'], result.effect.cumulative().value)
    assert_almost_equal(te['se'], result.effect.cumulative().se)


def test_vdid_zero_column():
    panel = california99
    panel.data['outcome']['VOID'] = 0.0
    panel.data['intervention']['VOID'] = 0

    panel.data['outcome']['VOID_T'] = 0.0
    panel.data['intervention']['VOID_T'] = panel.data['intervention']['California']

    estimator = DID()
    result = estimator.fit(california99)
    estimator.error(result, JackKnife())

    df = (CaliforniaProp99()
          .df()
          .pipe(lambda dx: dx.assign(treatment=dx['State'].isin(dx.query('treated == 1')['State'].unique())))
          .pipe(lambda dx: dx.assign(post=dx['Year'].isin(dx.query('treated == 1')['Year'].unique())))
          .assign(total='total')
          )

    treatment = df.groupby('treatment')['State'].unique()
    treatment[False] = list(treatment[False]) + ['VOID']
    treatment[True] = list(treatment[True]) + ['VOID_T']
    dims = dict(treatment=treatment)

    dte = vdid_panel(df, ['total'], 'PacksPerCapita', 'Year', 'State', dims=dims)
    te = dte['cum'].loc['total', 'PacksPerCapita']

    assert_almost_equal(te['te'], result.effect.cumulative().value)
    assert_almost_equal(te['se'], result.effect.cumulative().se)

