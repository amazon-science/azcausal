from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result
from azcausal.estimators.panel.did import did_simple


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

    def fit(self, panel, numerator=None, denominator=None, **kwargs):

        # if the effect on the numerator is not provided already, run the estimator
        if numerator is None:
            numerator = self.estimator.fit(panel.select(self.numerator), **kwargs)

        # if the effect on the denominator is not provided already, run the estimator
        if denominator is None:
            denominator = self.estimator.fit(panel.select(self.denominator), **kwargs)

        # check if the did estimates are available in the estimators
        assert 'did' in numerator.effect.data, "The numerator needs to have the DID estimates to calculate the ratio."
        assert 'did' in denominator.effect.data, "The denominator needs to have the DID estimates to calculate the ratio."

        # calculate the ratio effect using DID
        did = did_ratio(numerator.effect['did'], denominator.effect['did'])

        att = Effect(did["att"], observed=did["post_treat"], multiplier=panel.n_interventions(), data=did, name="ATT")
        return Result(dict(att=att), panel=panel, data=dict(numerator=numerator, denominator=denominator),
                      estimator=self)

    def refit(self, result, **kwargs):

        def f(panel):
            res = result.data['numerator']
            estimator = res.estimator
            numerator = estimator.refit(res, **kwargs)(panel.select(self.numerator))

            res = result.data['denominator']
            estimator = res.estimator
            denominator = estimator.refit(res, **kwargs)(panel.select(self.denominator))

            return self.fit(panel, numerator=numerator, denominator=denominator)

        return f


# calculate the ratio given two effects from DID
def did_ratio(numerator, denominator):

    pre_treat = numerator['pre_treat'] / denominator['pre_treat']
    pre_contr = numerator['pre_contr'] / denominator['pre_contr']
    post_treat = numerator['post_treat'] / denominator['post_treat']
    post_contr = numerator['post_contr'] / denominator['post_contr']

    return did_simple(pre_contr, post_contr, pre_treat, post_treat)


# just here for now for testing.
# class DIDRatio(Estimator):
#
#     def __init__(self, numerator, denominator, did=DID(), **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.did = did
#         self.numerator = numerator
#         self.denominator = denominator
#
#     def fit(self, panel, **kwargs):
#         nom = panel.select(self.numerator)
#         denom = panel.select(self.denominator)
#
#         pre_treat = nanmean(nom.Y(pre=True, treat=True)) / nanmean(denom.Y(pre=True, treat=True))
#         pre_contr = nanmean(nom.Y(pre=True, contr=True)) / nanmean(denom.Y(pre=True, contr=True))
#
#         post_treat = nanmean(nom.Y(post=True, treat=True)) / nanmean(denom.Y(post=True, treat=True))
#         post_contr = nanmean(nom.Y(post=True, contr=True)) / nanmean(denom.Y(post=True, contr=True))
#
#         did = did_simple(pre_contr, post_contr, pre_treat, post_treat)
#
#         data = dict()
#         att = Effect(did["att"], observed=did["post_treat"], multiplier=panel.n_interventions(), data=data, name="ATT")
#         return Result(dict(att=att), panel=panel, estimator=self)
