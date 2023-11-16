import numpy as np

from azcausal.core.error import Bootstrap
from azcausal.core.panel import Panel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.fuw import FixedUnitWeightsEstimator
from azcausal.util import to_matrices

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().load()

    # convert to matrices where the index represents each Year (time) and each column a state (unit)
    data = to_matrices(df, "Year", "State", "PacksPerCapita", "treated")

    # if there are nan values it was not balanced in the first place.
    assert np.isnan(data['treated'].values).sum() == 0, "The panel is not balanced."

    # create a panel object to access observations conveniently
    panel = Panel(outcome='PacksPerCapita', intervention='treated', data=data)

    # initialize an estimator object
    estimator = FixedUnitWeightsEstimator()

    # run the estimator
    result = estimator.fit(panel)

    # run the error validation method
    method = Bootstrap(n_samples=100)
    estimator.error(result, method)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
