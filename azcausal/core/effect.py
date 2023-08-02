from copy import copy

import numpy as np
import scipy

from azcausal.core.formating import DefaultFormat
from azcausal.core.output import Output


class Effect:

    def __init__(self,
                 value,
                 se=np.nan,
                 observed=None,
                 multiplier=None,
                 by_time=None,
                 by_unit=None,
                 treatment=1,
                 data=None,
                 name=None,
                 conf=95,
                 format=DefaultFormat()) -> None:

        super().__init__()
        self.value = value
        self.se = se
        self.observed = observed
        self.multiplier = multiplier

        self.conf = conf
        self.treatment = treatment
        self.by_time = by_time
        self.by_unit = by_unit
        self.data = data

        self.name = name
        self.format = format

    @property
    def counter_factual(self):
        if self.observed is not None:
            return self.observed - self.value

    def cumulative(self, multiplier=None, name="Cumulative"):
        if multiplier is None:
            multiplier = self.multiplier
        return self.multiply(multiplier, name=name)

    def percentage(self, name="Percentage", counter_factual=None):
        if counter_factual is None:
            counter_factual = self.counter_factual
        return self.multiply(100 / np.abs(counter_factual), name=name)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.data:
            return self.data[key]
        else:
            raise Exception(f"Key {key} not found.")

    def ci(self, conf=None):
        if self.value is not None and self.se is not None:
            if conf is None:
                conf = self.conf

            return scipy.stats.norm.interval(conf / 100, loc=self.value, scale=self.se)

    def set_name(self, name):
        self.name = name
        return self

    def __str__(self):
        return f"{self.value} (se: {self.se})"

    def multiply(self, s, name=None):
        total = copy(self)
        total.name = name

        # multiply all the import values
        total.value *= s
        total.se *= s
        total.observed *= s

        return total

    def sign(self, conf=None, no_effect='+/-'):
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

    def summary(self, title=None, conf=None, **kwargs):

        if title is None:
            title = self.name

        if conf is None:
            conf = self.conf

        out = Output()
        out.text(title, align="center")

        has_se = not (self.se is None or np.isnan(self.se))

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


