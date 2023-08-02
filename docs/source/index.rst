

Home
====================================


.. toctree::
   :hidden:

   self
   introduction/introduction
   estimators/estimators
   tutorials/tutorials






Causal inference is an important component of the experiment evaluation. We highly recommend to have a look at the open-source
book: `Causal Inference for The Brave and True <https://matheusfacure.github.io/python-causality-handbook/landing-page.html>`_.


Currently, azcausal provides two well-known and widely used causal inference methods: Difference-in-Difference (DID) and
Synthetic Difference-in-Difference (SDID). Moreover, error estimates via Placebo, Boostrap, or JackKnife are available.
Given the result of an estimator printing the summary looks as follows:


.. code:: bash

    ╭──────────────────────────────────────────────────────────────────────────────╮
    |                               CaliforniaProp99                               |
    ├==============================================================================┤
    |                                    Panel                                     |
    |  Time Periods: 31 (19/12)                                  total (pre/post)  |
    |  Units: 39 (38/1)                                       total (contr/treat)  |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                     ATT                                      |
    |  Effect (±SE): -15.60 (±7.7087)                                              |
    |  Confidence Interval (95%): [-30.71 , -0.495030]                        (-)  |
    |  Observed: 60.35                                                             |
    |  Counter Factual: 75.95                                                      |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                  Percentage                                  |
    |  Effect (±SE): -20.54 (±10.15)                                               |
    |  Confidence Interval (95%): [-40.44 , -0.651752]                        (-)  |
    |  Observed: 79.46                                                             |
    |  Counter Factual: 100.00                                                     |
    ├──────────────────────────────────────────────────────────────────────────────┤
    |                                  Cumulative                                  |
    |  Effect (±SE): -187.25 (±92.50)                                              |
    |  Confidence Interval (95%): [-368.55 , -5.9404]                         (-)  |
    |  Observed: 724.20                                                            |
    |  Counter Factual: 911.45                                                     |
    ╰──────────────────────────────────────────────────────────────────────────────╯


Moreover azcausal supports plotting the impact as well:

.. image:: https://github.com/amazon-science/azcausal/blob/d176af3d41144f5c5fa4d8ef31f5484c4953c6b7/docs/source/images/sdid.png?raw=true



Estimators
********************************************************************************


- **Difference-in-Difference (DID):** Simple implementation of the well-known Difference-in-Difference estimator.
- **Synthetic Difference-in-Difference (SDID):** Arkhangelsky, Dmitry Athey, Susan Hirshberg, David A. Imbens, Guido W. Wager, Stefan Synthetic Difference-in-Differences American Economic Review 111 12 4088-4118 2021 10.1257/aer.20190159 https://www.aeaweb.org/articles?id=10.1257/aer.20190159. Implementation based on https://synth-inference.github.io/synthdid/


Contact
********************************************************************************

Feel free to contact me if you have any questions:

| `Julian Blank <http://julianblank.com>`_  (blankjul [at] amazon.com)
| Amazon.com
| Applied Scientist, Amazon
| 410 Terry Ave N, Seattle 98109, WA.



