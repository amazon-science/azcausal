import numpy as np

from azcausal.core.error import Placebo
from azcausal.core.panel import Panel
from azcausal.core.parallelize import Pool
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID
from azcausal.util import to_matrices

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().load()

    # convert to matrices where the index represents each Year (time) and each column a state (unit)
    outcome, intervention = to_matrices(df, "Year", "State", "PacksPerCapita", "treated")

    # if there are nan values it was not balanced in the first place.
    assert np.isnan(outcome.values).sum() == 0, "The panel is not balanced."

    # create a panel object to access observations conveniently
    pnl = Panel(outcome, intervention)

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID()

    # run the estimator
    estm = estimator.fit(pnl)
    print("Average Treatment Effect on the Treated (ATT):", estm["att"])

    # create a process pool for parallelization
    pool = Pool(mode="processes", progress=True)

    # run the error validation method
    method = Placebo(n_samples=50)
    err = estimator.error(estm, method, parallelize=pool)

    print("Standard Error (se):", err["se"])
    print("Error Confidence Interval (90%):", err["CI"]["90%"])
