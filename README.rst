azcausal: Causal Inference in Python
====================================================================

Causal inference is an important component of the experiment evaluation. We highly recommend to have a look at the open-source
book: `Causal Inference for The Brave and True <https://matheusfacure.github.io/python-causality-handbook/landing-page.html>`_

Please find the software documentation here: https://amazon-science.github.io/azcausal/latest/

Currently, azcausal provides two well-known and widely used causal inference methods: Difference-in-Difference (DID) and
Synthetic Difference-in-Difference (SDID). Moreover, error estimates via Placebo, Boostrap, or JackKnife are available.


.. _Installation:

Installation
********************************************************************************


To install the current release, please execute:

.. code:: bash

    pip install git+https://github.com/amazon-science/azcausal.git


.. _Usage:

Usage
********************************************************************************


.. code:: python

    from azcausal.core.error import JackKnife
    from azcausal.core.panel import CausalPanel
    from azcausal.data import CaliforniaProp99
    from azcausal.estimators.panel.sdid import SDID
    from azcausal.util import to_panels


    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().df()

    # create the panel data from the frame and define the causal types
    data = to_panels(df, 'Year', 'State', ['PacksPerCapita', 'treated'])
    ctypes = dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    # initialize the panel
    panel = CausalPanel(data).setup(**ctypes)

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID()

    # run the estimator
    result = estimator.fit(panel)

    # run the error validation method
    estimator.error(result, JackKnife())

    # plot the results
    estimator.plot(result)

    # print out information about the estimate
    print(result.summary(title="CaliforniaProp99"))


.. code:: bash

    ╭──────────────────────────────────────────────────────────────────────────────╮
    |                               CaliforniaProp99                               |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                    Panel                                     |
    |  Time Periods: 31 (19/12)                                  total (pre/post)  |
    |  Units: 39 (38/1)                                       total (contr/treat)  |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                     ATT                                      |
    |  Effect (±SE): -15.60 (±2.9161)                                              |
    |  Confidence Interval (95%): [-21.32 , -9.8884]                          (-)  |
    |  Observed: 60.35                                                             |
    |  Counter Factual: 75.95                                                      |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                  Percentage                                  |
    |  Effect (±SE): -20.54 (±3.8393)                                              |
    |  Confidence Interval (95%): [-28.07 , -13.02]                           (-)  |
    |  Observed: 79.46                                                             |
    |  Counter Factual: 100.00                                                     |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                  Cumulative                                  |
    |  Effect (±SE): -187.25 (±34.99)                                              |
    |  Confidence Interval (95%): [-255.83 , -118.66]                         (-)  |
    |  Observed: 724.20                                                            |
    |  Counter Factual: 911.45                                                     |
    ╰──────────────────────────────────────────────────────────────────────────────╯

.. image:: docs/source/images/sdid.png

.. _Estimators:

Estimators
********************************************************************************


- **Difference-in-Difference (DID):** Simple implementation of the well-known Difference-in-Difference estimator.
- **Synthetic Difference-in-Difference (SDID):** Arkhangelsky, Dmitry Athey, Susan Hirshberg, David A. Imbens, Guido W. Wager, Stefan Synthetic Difference-in-Differences American Economic Review 111 12 4088-4118 2021 10.1257/aer.20190159 https://www.aeaweb.org/articles?id=10.1257/aer.20190159. Implementation based on https://synth-inference.github.io/synthdid/

.. _Contact:

Contact
********************************************************************************

Feel free to contact me if you have any questions:

| `Julian Blank <http://julianblank.com>`_  (blankjul [at] amazon.com)
| Amazon.com
| Applied Scientist, Amazon
| 410 Terry Ave N, Seattle 98109, WA.


