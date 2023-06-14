from azcausal.core.error import JackKnife
from azcausal.core.panel import Panel
from azcausal.estimators.panel.did import EventStudy, DID
from azcausal.util import zeros_like, to_matrix
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID

import numpy as np

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

    print("Event Study with `pre=0` should always match DiD (but provide impact over time)")

    estm_did = DID().fit(pnl)
    print("[DID] Average Treatment Effect on the Treated (ATT):", estm_did["att"])

    estm_event = EventStudy(n_pre=0).fit(pnl)
    print("[Event Study] Average Treatment Effect on the Treated (ATT):", estm_event["att"])

    delta = (estm_did["data"]["att"].dropna().values - estm_event["data"]["att"].values)
    print("Difference in Estimations", np.abs(delta).sum())

    # now perform the event study
    estimator = EventStudy(n_pre=None, exclude=-1)
    estm = estimator.fit(pnl)

    # show the results in a plot
    estimator.plot(estm)
