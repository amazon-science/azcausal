from azcausal.core.error import JackKnife
from azcausal.core.panel import CausalPanel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.util import to_panels

if __name__ == '__main__':
    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().df()

    # create the panel data from the frame and define the causal types
    data = to_panels(df, 'Year', 'State', ['PacksPerCapita', 'treated'])
    ctypes = dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    # initialize the panel
    panel = CausalPanel(data).setup(**ctypes)

    # initialize an estimator object
    estimator = DID()

    # run the estimator
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())

    # plot the results
    estimator.plot(result)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
