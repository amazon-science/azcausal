import numpy as np

from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.ratio import Ratio
from azcausal.estimators.panel.sdid import SDID


def test_ratio_panel():
    cdf = CaliforniaProp99().cdf()
    cdf['random'] = np.random.uniform(size=len(cdf))
    panel = cdf.to_panel()

    estimator = SDID()
    result = Ratio(estimator, 'outcome', 'random').fit(panel)
    print(result.summary())

    assert result.effect.value is not None
