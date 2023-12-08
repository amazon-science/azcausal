import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
from linearmodels.panel.utility import AbsorbingEffectError


class CausalRegression(object):

    def __init__(self, on="intervention", get=None, mode='PanelOLS',
                 time_effects=True, unit_effects=True, fixed_effects=None):
        self.on = on
        self.get = get if get is not None else on
        self.mode = mode

        if fixed_effects is not None:
            if fixed_effects:
                time_effects = True
                unit_effects = True
            else:
                time_effects = False
                unit_effects = False

        self.time_effects = time_effects
        self.unit_effects = unit_effects

    def fit(self, cdf, weights=None):

        param = np.nan
        se = np.nan
        summary = None
        result = None

        formula = f'outcome ~ {self.on}'

        try:

            if self.mode == "PanelOLS":

                if self.time_effects:
                    formula += f' + TimeEffects'
                if self.unit_effects:
                    formula += f' + EntityEffects'

                data = pd.DataFrame(cdf).set_index(['unit', 'time'])
                if weights is not None:
                    weights = data[weights]
                res = PanelOLS.from_formula(formula=formula, data=data, weights=weights).fit(low_memory=True)

                param = res.params[self.get]
                se = res.std_errors[self.get]
                summary = res.summary
                result = res

            elif self.mode == "OLS":
                if self.time_effects:
                    formula += f' + C(time)'
                if self.unit_effects:
                    formula += f' + C(unit)'

                if weights is not None:
                    res = smf.wls(formula=formula, data=cdf.reset_index(), weights=cdf[weights]).fit()
                else:
                    res = smf.ols(formula=formula, data=cdf.reset_index()).fit()

                param = res.params[self.get]
                se = res.bse[self.get]
                summary = res.summary()
                result = res

        except AbsorbingEffectError:
            pass

        return dict(param=param, se=se, summary=summary, result=result, formula=formula)
