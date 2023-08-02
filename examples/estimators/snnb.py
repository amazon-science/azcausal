from azcausal.core.error import Placebo
from azcausal.core.parallelize import Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.snnb import SNNB

if __name__ == '__main__':

    # create the panel to be fed into the estimator
    panel = CaliforniaProp99().panel()

    # initialize an estimator object
    estimator = SNNB()

    # run the estimator
    result = estimator.fit(panel)

    # plot the results
    estimator.plot(result, title="CaliforniaProp99")

    # create a process pool for parallelization
    parallelize = Joblib()

    # run the error validation method
    method = Placebo(n_samples=11)
    estimator.error(result, method, parallelize=parallelize)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))

