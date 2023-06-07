azcausal: Causal Inference in Python
====================================================================

Causal inference is an important component of the experiment evaluation. We highly recommend to have a look at the open-source
book: `Causal Inference for The Brave and True <https://matheusfacure.github.io/python-causality-handbook/landing-page.html>`_


Currently, azcausal provides two well-known and widely used causal inference methods: Difference-in-Difference (DID) and
Synthetic Difference-in-Difference (SDID). Moreover, error estimates via Placebo, Boostrap, or JackKnife are available.


.. _Installation:

Installation
********************************************************************************

The official release is always available at PyPi:

.. code:: bash

    pip install -U azcausal


For the current developer version:

.. code:: bash

    pip install git+ssh://git@github.com/amazon-science/azcausal.git


.. _Usage:

Usage
********************************************************************************


.. code:: python

    from azcausal.core.error import JackKnife
    from azcausal.core.panel import Panel
    from azcausal.util import zeros_like, to_matrix
    from azcausal.data import CaliforniaProp99
    from azcausal.estimators.panel.sdid import SDID


    # load an example data set with the columns Year, State, PacksPerCapita, treated.
    df = CaliforniaProp99().load()

    # convert to matrices where the index represents each Year (time) and each column a state (unit)
    outcome = to_matrix(df, "Year", "State", "PacksPerCapita", fillna=0.0)

    # the time when the intervention started
    start_time = df.query("treated == 1")["Year"].min()

    # the units that have been treated
    treat_units = list(df.query("treated == 1")["State"].unique())

    # create the treatment matrix based on the information above
    treatment = zeros_like(outcome)
    treatment.loc[start_time:, treatment.columns.isin(treat_units)] = 1

    # create a panel object to access observations conveniently
    pnl = Panel(outcome, treatment)

    # initialize an estimator object, here synthetic difference in difference (sdid)
    estimator = SDID()

    # run the estimator
    estm = estimator.fit(pnl)
    print("Average Treatment Effect on the Treated (ATT):", estm["att"])

    # show the results in a plot
    estimator.plot(estm, trend=True, sc=True)

    # run an error validation method
    method = JackKnife()
    err = estimator.error(estm, method)

    print("Standard Error (se):", err["se"])
    print("Error Confidence Interval (90%):", err["CI"]["90%"])


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


