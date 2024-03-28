from azcausal.core.panel import CausalPanel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.fsdid import FSDID
from azcausal.util import to_panels

if __name__ == '__main__':
    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().df()

    # create the panel data from the frame and define the causal types
    data = to_panels(df, 'Year', 'State', ['PacksPerCapita', 'treated'])
    ctypes = dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    # initialize the panel
    panel = CausalPanel(data).setup(**ctypes)

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = FSDID()

    # run the estimator
    result = estimator.fit(panel)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))
