from azcausal.core.error import Placebo
from azcausal.core.parallelize import Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID

if __name__ == '__main__':

    panel = CaliforniaProp99().panel()

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
    print(result.summary(title="CaliforniaProp99", conf=90))

