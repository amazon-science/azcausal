from azcausal.core.error import Placebo
from azcausal.core.panel import Panel
from azcausal.core.parallelize import Pool
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.ssnbiclustering import SNNBiclustering
from azcausal.util import to_matrix, zeros_like

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().load()

    # convert to matrices where the index represents each Year (time) and each column a state (unit)
    outcome = to_matrix(df, "Year", "State", "PacksPerCapita", fillna=0.0)

    # the time when the intervention started
    start_time = df.query("treated == 1")["Year"].min()

    # the units that have been treated
    treat_units = list(df.query("treated == 1")["State"].unique())

    # create the treatment matrix based on the information above
    intervention = zeros_like(outcome)
    intervention.loc[start_time:, intervention.columns.isin(treat_units)] = 1

    pnl = Panel(outcome, intervention)

    # initialize an estimator object, here difference in difference (did)
    estimator = SNNBiclustering()

    # run the estimator
    estm = estimator.fit(pnl)
    print("Average Treatment Effect on the Treated (ATT):", estm["att"])

    # plot the results
    estimator.plot(estm, title="CaliforniaProp99")

    # create a process pool for parallelization
    pool = Pool(mode="processes", progress=True)

    # run the error validation method
    method = Placebo(n_samples=50)
    err = estimator.error(estm, method, parallelize=pool)

    print("Standard Error (se):", err["se"])
    print("Error Confidence Interval (90%):", err["CI"]["90%"])
