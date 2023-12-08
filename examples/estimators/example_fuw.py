from azcausal.core.error import JackKnife
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.fuw import FixedUnitWeightsEstimator

if __name__ == '__main__':

    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    panel = CaliforniaProp99().panel()

    # initialize an estimator object
    estimator = FixedUnitWeightsEstimator()

    # run the estimator
    result = estimator.fit(panel)

    # run the error validation method
    method = JackKnife()
    estimator.error(result, method)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
