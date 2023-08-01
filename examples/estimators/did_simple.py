from azcausal.core.error import Placebo
from azcausal.core.panel import Panel
from azcausal.core.parallelize import Pool, Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.util import to_matrix, intervention_from_outcome

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().load()

    # convert to matrices where the index represents each Year (time) and each column a state (unit)
    outcome = to_matrix(df, "Year", "State", "PacksPerCapita", fillna=0.0)

    # the time when the intervention started
    start_time = df.query("treated == 1")["Year"].min()

    # the units that have been treated
    treat_units = list(df.query("treated == 1")["State"].unique())

    # create the intervention matrix
    intervention = intervention_from_outcome(outcome, start_time, treat_units)

    panel = Panel(outcome, intervention)

    # initialize an estimator object, here difference in difference (did)
    estimator = DID()

    # run the estimator
    result = estimator.fit(panel)
    print("Average Treatment Effect on the Treated (ATT):", result["att"].value)

    # plot the results
    estimator.plot(result, title="CaliforniaProp99")

    # create a process pool for parallelization
    pool = Joblib(n_jobs=5, progress=True)

    # run the error validation method
    method = Placebo(n_samples=101)
    estimator.error(result, method, parallelize=pool)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))

