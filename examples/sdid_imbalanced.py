from azcausal.core.error import JackKnife
from azcausal.core.panel import Panel
from azcausal.util import zeros_like, to_matrix
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID

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

    # create a panel object to access observations conveniently
    pnl = Panel(outcome, intervention)

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID()

    # run the estimator
    estm = estimator.fit(pnl)
    print("Average Treatment Effect on the Treated (ATT):", estm["att"])

    # show the results in a plot
    estimator.plot(estm, trend=True, sc=True)

    # run an error validation method
    method = JackKnife()
    err = estimator.error(estm, method)

    print("Standard Error (se):", err["se"])
    print("Error Confidence Interval (90%):", err["CI"]["90%"])
