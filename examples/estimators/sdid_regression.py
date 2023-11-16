from azcausal.core.error import JackKnife
from azcausal.core.panel import Panel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID
from azcausal.util import to_matrix, intervention_from_outcome

if __name__ == '__main__':

    # create a panel object to access observations conveniently
    panel = CaliforniaProp99().panel()

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID(regression=True)

    # run the estimator
    result = estimator.fit(panel)

    # show the results in a plot
    estimator.plot(result, CF=True, C=True)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
