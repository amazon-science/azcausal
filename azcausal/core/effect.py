from __future__ import annotations

from copy import copy

import numpy as np
import pandas as pd
import scipy

from azcausal.core.formating import DefaultFormat, Format
from azcausal.core.output import Output


class Effect:

    def __init__(self,
                 value: float,
                 se: float = np.nan,
                 observed: float = None,
                 dof: float = 1,
                 scale: float = None,
                 by_time: pd.DataFrame = None,
                 by_unit: pd.DataFrame = None,
                 treatment: int = 1,
                 data: dict = None,
                 name: str = None,
                 conf: float = 95,
                 format: Format = DefaultFormat()) -> None:
        """
        The `Effect` object summarizes the treatment effects returned by an estimator.

        Parameters
        ----------
        value
            The value of the effect which will be in most cases either ATT or ATE.

        se
            The standard error of this effect.

        observed
            The observed value during the treatment period.

        dof
            The degree of freedom (can be used to calculate the confidence intervals)

        scale
            We usually have an overage value for the treatment effects. If we want to aggregate to cumulative
            the scale is stored here.

        by_time
            The effect over time periods.

        by_unit
            The effect by units.

        treatment
            The treatment type in the intervention matrix.

        data
            Any additional data that need to be stored from the estimator.

        name
            A name can be passed if desired (e.g. type of the effect)

        conf
            The confidence when the intervals are calculated, e.g. 95 corresponds to the 95% confidence interval.

        format
            The formatting that should be used when printing floats in the summary.
        """

        super().__init__()
        self.value = value
        self.se = se
        self.dof = dof
        self.observed = observed
        self.scale = scale

        self.conf = conf
        self.treatment = treatment
        self.by_time = by_time
        self.by_unit = by_unit
        self.data = data if data is not None else dict()

        self.name = name
        self.format = format

    @property
    def counter_factual(self):
        """
        Calculate the counterfactual by: observed - value.

        Note: This is stored as a property here to change with observed and value.

        Returns
        -------
        counter_factual
            The counterfactual of the treatment units during the experiment.

        """
        if self.observed is not None:
            return self.observed - self.value

    def cumulative(self,
                   scale: float = None,
                   name: str = "Cumulative") -> Effect:
        """
        This method the cumulative effect given an average.

        Parameters
        ----------
        scale
            The scale can be overwritten otherwise the one set by the estimator is used.

        name
            The name of the new effect.

        Returns
        -------
        effect
            The effect having the cumulative as value.

        """
        if scale is None:
            scale = self.scale

        assert scale is not None, "To get the cumulative effect a scale is needed."

        return self.multiply(scale, name=name)

    def relative(self,
                 counter_factual=None,
                 name="Relative"):
        if counter_factual is None:
            counter_factual = self.counter_factual
        assert counter_factual is not None, "To get the percentage effect counter factual is needed."
        return self.multiply(1 / np.abs(counter_factual), name=name)

    def percentage(self,
                   counter_factual=None,
                   name="Percentage"):
        """
        This method returns the effect based on relative impact.

        Parameters
        ----------
        counter_factual
            The counter-factual which is used to calculate the percentage.

        name
            The name of the new effect.

        Returns
        -------
        effect
            The effect having the percentage as value.

        """
        rel_effect = self.relative(counter_factual=counter_factual)
        return rel_effect.multiply(100, name=name)

    def multiply(self, s: float, name=None):
        """
        This method multiplies this effect by a coefficient.
        Parameters
        ----------
        s
            The coefficient.
        name
            The new name of the effect.

        Returns
        -------
        effect
            The resulting effect.

        """
        return Effect(value=self.value * s, se=self.se * s, observed=self.observed * s, name=name,
                      conf=self.conf, format=self.format, dof=self.dof, treatment=self.treatment)

    def ci(self, conf: float = None, stat_test: str = "z") -> tuple:
        """
        Return the confidence intervals (ci).

        Parameters
        ----------
        conf
            The confidence, e.g. 95 represents the 95% confidence interval.

        stat_test
            The statistical test that should be used.

        Returns
        -------
        ci
            Returns a tuple (lower, upper) representing the confidence intervals

        """
        value, se, dof = self.value, self.se, self.dof
        if conf is None:
            conf = self.conf

        if value is not None and se is not None:

            if np.allclose(se, 0.0):
                return value, value

            else:

                if stat_test == 'z':
                    ci = scipy.stats.norm.interval(conf / 100, loc=value, scale=se)

                elif stat_test == 't':
                    assert dof is not None, "The t-test needs to degrees of freedom."
                    ci = scipy.stats.t.interval(conf / 100, dof, loc=value, scale=se)
                else:
                    raise Exception("Use either z or t test for the confidence interval.")

                return ci

    def sign(self, conf: float = None, no_effect: str = '+/-'):
        """
        Returns the sign of the estimate given a confidence interval.
            '+': Positive impact (confidence level lower bound is positive)
            '+/-': Statistically insignificant
            '-': Negative Impact (confidence level upper bound is negative)

        Parameters
        ----------
        conf
            The confidence interval.

        no_effect
            The sign to use if statistically insignificant

        Returns
        -------
        sign
            A string representing the sign of the estimate.

        """
        ci = self.ci(conf=conf)

        if ci is None:
            return np.nan
        else:
            l, u = ci
            if u < 0:
                return '-'
            elif l > 0:
                return '+'
            else:
                return no_effect

    def copy(self, **kwargs):
        obj = copy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def to_dict(self, prefix='', conf=None):

        if conf is None:
            conf = self.conf

        vals = [('avg', self),
                ('rel', self.relative()),
                ('perc', self.percentage()),
                ('cum', self.cumulative()),
                ]

        entry = dict(effect=self, sign=self.sign(conf=conf))

        for name, effect in vals:
            entry[f"{name}_te"] = effect.value
            entry[f"{name}_ci_lb"], entry[f"{name}_ci_ub"] = effect.ci(conf=conf)

        return {f"{prefix}{k}": v for k, v in entry.items()}

    def summary(self, title: str = None, conf: float = None, **kwargs):
        """
        The `Summary` of this effect to conveniently print it out..

        Parameters
        ----------
        title
            Provide a title otherwise the name of this effect will be used.

        conf
            The confidence level

        Returns
        -------
        summary
            A `Summary` object.

        """

        if title is None:
            title = self.name

        if conf is None:
            conf = self.conf

        out = Output()
        out.text(title, align="center")

        has_se = self.se is not None and not np.isnan(self.se)

        if not has_se:
            out.text(f"Effect: {self.format(self.value)}")
        else:
            out.text(f"Effect (\u00B1SE): {self.format(self.value)} (\u00B1{self.format(self.se)})")

            if conf:
                a = conf
                l, u = self.ci(a)
                out.texts(
                    [f"Confidence Interval ({conf}%): [{self.format(l)} , {self.format(u)}]", f"({self.sign(a)})"])

        if self.observed is not None:
            out.text(f"Observed: {self.format(self.observed)}")
            out.text(f"Counter Factual: {self.format(self.counter_factual)}")

        return out

    def __str__(self):
        return f"{self.value} (se: {self.se})"

    # a convenience function to access custom data from the estimator
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.data:
            return self.data[key]
        else:
            raise Exception(f"Key {key} not found.")


def get_true_effect(cdf):
    scale = cdf.n_interventions()
    observed = cdf.observed()
    att = cdf['te'].values.sum() / scale
    se = 0.0
    return Effect(att, se=se, observed=observed, scale=scale)
