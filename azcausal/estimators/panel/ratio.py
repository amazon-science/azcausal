from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result
from azcausal.estimators.panel.did import did_equation


class Ratio(Estimator):

    def __init__(self, estimator: Estimator,
                 numerator: str,
                 denominator: str,
                 **kwargs) -> None:
        """

        Given panel data with two different outcomes, this method estimates their ratio.

        Parameters
        ----------
        estimator
            An estimator to predict an effect (needs to store DID data).

        numerator
            A string representing the numerator in the panel data.

        denominator
            A string representing the denominator in the panel data.

        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.numerator = numerator
        self.denominator = denominator

    def fit(self, cdf, numerator=None, denominator=None, **kwargs):

        # if the effect on the numerator is not provided already, run the estimator
        if numerator is None:
            cdx = cdf.assign(outcome=lambda dd: dd[self.numerator])
            numerator = self.estimator.fit(cdx, **kwargs)

        # if the effect on the denominator is not provided already, run the estimator
        if denominator is None:
            cdx = cdf.assign(outcome=lambda dd: dd[self.denominator])
            denominator = self.estimator.fit(cdx, **kwargs)

        # check if the did estimates are available in the estimators
        assert 'did' in numerator.effect.data, "The numerator needs to have the DID estimates to calculate the ratio."
        assert 'did' in denominator.effect.data, "The denominator needs to have the DID estimates to calculate the ratio."

        # calculate the ratio effect using DID
        did = did_ratio(numerator.effect['did'], denominator.effect['did'])

        att = Effect(did["att"], observed=did["post_treat"], scale=cdf.n_interventions(), data=did, name="ATT")
        info = dict(numerator=numerator, denominator=denominator)
        return Result(dict(att=att), info=info, data=cdf, estimator=self)

    def refit(self, result, **kwargs):

        def f(panel):
            res = result.info['numerator']
            estimator = res.estimator
            numerator = estimator.refit(res, **kwargs)(panel.select(self.numerator))

            res = result.info['denominator']
            estimator = res.estimator
            denominator = estimator.refit(res, **kwargs)(panel.select(self.denominator))

            return self.fit(panel, numerator=numerator, denominator=denominator)

        return f


def did_ratio(numerator, denominator):
    pre_treat = numerator['pre_treat'] / denominator['pre_treat']
    pre_contr = numerator['pre_contr'] / denominator['pre_contr']
    post_treat = numerator['post_treat'] / denominator['post_treat']
    post_contr = numerator['post_contr'] / denominator['post_contr']

    return did_equation(pre_contr, post_contr, pre_treat, post_treat)
