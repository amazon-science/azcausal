import numpy as np

from azcausal.core.error import Placebo
from azcausal.core.panel import Panel
from azcausal.core.parallelize import Pool, Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID
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

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID()

    # run the estimator
    result = estimator.fit(panel)

    # create a thread pool for parallelization
    pool = Joblib(progress=True)

    # run the error validation method
    method = Placebo(n_samples=11)
    estimator.error(result, method, parallelize=pool)

    # finally plot the results
    estimator.plot(result, CF=True, C=True)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
