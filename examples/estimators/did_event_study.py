import numpy as np

from azcausal.core.panel import Panel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import EventStudy, DID
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

    # create a panel object to access observations conveniently
    panel = Panel(outcome, intervention)

    print("Event Study with `pre=0` should always match DiD (but provide impact over time)")

    did = DID().fit(panel)
    print("[DID] Average Treatment Effect on the Treated (ATT):", did["att"].value)

    event = EventStudy(n_pre=0).fit(panel)
    print("[Event Study] Average Treatment Effect on the Treated (ATT):", did["att"].value)

    desired = did.effect.by_time.query("W == 1")["att"].values
    actual = event.effect.by_time["att"].values
    delta = np.abs(desired - actual)
    print("Difference in Estimations", delta.sum())

    # now perform the event study
    estimator = EventStudy(n_pre=None, exclude=-1)
    result = estimator.fit(panel)

    # show the results in a plot
    estimator.plot(result)
