from azcausal.core.panel import Panel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DIDRegressor
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

    # create the intervention matrix based on the information above
    intervention = intervention_from_outcome(outcome, start_time, treat_units)

    # create the panel data structure
    panel = Panel(outcome, intervention)

    # initialize an estimator object, here difference in difference (did)
    estimator = DIDRegressor()

    # fit the estimator
    result = estimator.fit(panel)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))

    from azcausal.core.panel import Panel
    from azcausal.data import CaliforniaProp99
    from azcausal.estimators.panel.did import DIDRegressor
    from azcausal.util import to_matrix, intervention_from_outcome

    dy = df.rename(columns=dict(State='unit', Year='time', PacksPerCapita='outcome', treated='intervention'))

    # initialize an estimator object, here difference in difference (did)
    estimator = DIDRegressor()

    # fit the estimator
    result = estimator.fit(dy)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))



