import numpy as np

from azcausal.data import CaliforniaProp99
from azcausal.estimators.event_study import EventStudy
from azcausal.estimators.panel.did import DID

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    cdf = CaliforniaProp99().cdf()

    print("Event Study with `pre=0` should always match DiD (but provide impact over time)")

    did = DID().fit(cdf.to_panel())
    print("[DID] Average Treatment Effect on the Treated (ATT):", did["att"].value)

    event = EventStudy(n_pre=0).fit(cdf)
    print("[Event Study] Average Treatment Effect on the Treated (ATT):", did["att"].value)

    desired = did.effect.by_time.query("post == 1")["att"].values
    actual = event.effect.by_time["att"].values
    delta = np.abs(desired - actual)
    print("Difference in Estimations", delta.sum())

    # now perform the event study
    estimator = EventStudy(n_pre=None, exclude=-1)
    result = estimator.fit(cdf)

    # show the results in a plot
    estimator.plot(result)
