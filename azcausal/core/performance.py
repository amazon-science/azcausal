from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import kstest


def power(results: list,
          conf: float = 90):
    counts = Counter([result.effect.sign(conf=conf) for result in results])
    return {s: counts[s] / len(results) for s in ['+', '+/-', '-']}


def performance(results, mode='perc'):
    return pd.DataFrame(([perf_eval(result[mode]) for result in results]))


def perf_eval(e):
    pred_att = e['att']
    pred_se = e['se']
    true_att = e['true_att']

    if pred_se:
        pval = kstest([(true_att - pred_att / pred_se)], 'norm').pvalue
    else:
        pval = np.nan

    entry = {
        'ae': np.abs(pred_att - true_att),
        'se': (pred_att - true_att) ** 2,
        'pval': pval
    }

    return entry
