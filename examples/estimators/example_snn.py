from azcausal.core.error import Bootstrap
from azcausal.core.parallelize import Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.snn import SNN

if __name__ == '__main__':

    # create the panel to be fed into the estimator
    panel = CaliforniaProp99().panel()

    # initialize an estimator object
    estimator = SNN()

    # run the SNN
    result = estimator.fit(panel)

    # estimate the standard error
    estimator.error(result, Bootstrap(n_samples=31), parallelize=Joblib(progress=True))

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))

