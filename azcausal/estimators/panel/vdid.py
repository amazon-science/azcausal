from typing import Iterator, List

import numpy as np
import pandas as pd
import scipy
from numpy.random import RandomState


def format_human_readable(num):
    if num is None:
        return ""
    elif type(num) in [str]:
        return num
    elif num == np.inf:
        return "inf"
    else:
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def format_float(precision=2):

    def f(num):
        if num is None:
            return ""
        elif type(num) in [str]:
            return num
        else:
            return ('{:.' + str(precision) + 'f}').format(num)


def vdid_highlight(row):
    if 'sign' not in row:
        color = None
    elif row['sign'] == '+':
        color = 'lightgreen'
    elif row['sign'] == '-':
        color = 'lightcoral'
    else:
        color = 'lightgray'
    return [f'background-color: {color}' for _ in range(len(row))]


def set_to_obj(obj, k, v):
    setattr(obj, k, v)
    return obj


def vdid_ci(dx, conf):
    lb, ub = scipy.stats.norm.interval(conf / 100, loc=dx['te'], scale=(dx['se'] + 1e-16))
    return (dx
            .assign(lb=lb, ub=ub)
            .assign(sign=lambda dx: dx.apply(vdid_sign, axis=1))
            )


def vdid_multiply(df, scale, conf):
    return vdid_ci(df[['te', 'se']].multiply(scale, axis='index'), conf)


def group_by_index(dx):
    return dx.groupby(list(dx.index.names))


def vdid_avg_by(dx, label, col):
    counts = dx.reset_index().groupby(label)[col].nunique()
    avg = group_by_index(dx.droplevel(col, axis='index')).sum().multiply(1 / counts, axis='index', level=label)
    return avg, counts


def vdid_avg(dx, groups, counts=None):
    if counts is None:
        counts = dict()
    for label, col in groups:
        dx, counts[label] = vdid_avg_by(dx, label, col)
    return dx, counts


def vdid_did(dx):
    while isinstance(dx, pd.DataFrame):
        dx = dx[True] - dx[False]
    return dx


def vdid_take_by_column(dx, col):
    while isinstance(dx, pd.DataFrame):
        dx = dx[col]
    return dx


def vdid_jackknife():
    def sample(treatment: pd.Series) -> Iterator[pd.Series]:
        treat, contr = treatment[True], treatment[False]

        if len(contr) > 1:
            for i in range(len(contr)):
                yield pd.Series({True: treat, False: np.delete(contr, i)})

        if len(treat) > 1:
            for i in range(len(treat)):
                yield pd.Series({True: np.delete(treat, i), False: contr})

    def fit(dse: pd.DataFrame):
        n = len(dse)
        var = dse.var(ddof=1, numeric_only=True)
        return np.sqrt(((n - 1) / n) * (n - 1) * var)

    return sample, fit


def vdid_se(dx):
    n = len(dx)
    return np.sqrt((n - 1) / n) * dx.std(ddof=1, numeric_only=True)


def vdid_bootstrap_balanced(n_samples=1000, seed=1):
    def sample(treatment: pd.Series) -> Iterator[pd.Series]:
        for i in range(n_samples):
            yield treatment.map(lambda x: RandomState(seed + i).choice(x, size=len(x), replace=True))

    return sample, vdid_se


def vdid_bootstrap(n_samples=1000, seed=1):
    def sample(treatment: pd.Series) -> Iterator[pd.Series]:

        H = dict()
        for unit in treatment[True]:
            H[unit] = True
        for unit in treatment[False]:
            H[unit] = False
        H = pd.Series(H, name='treatment').rename_axis('region', axis='index')

        cnt = 0
        max_cnt = n_samples

        random_state = RandomState(seed)

        while cnt < max_cnt:

            ss = sorted(random_state.choice(list(H.keys()), size=len(H), replace=True))
            tt = H.loc[ss].to_frame().reset_index().groupby('treatment')['region'].apply(lambda x: list(x))

            # if we have at least one treatment and one control
            if tt.map(lambda x: len(x) > 0).all():
                yield tt
                cnt += 1

    return sample, vdid_se


def dot_by_columns(ds, columns, name, weights=None):
    if weights is None:
        weights = columns.map(lambda x: np.full(len(x), 1 / len(x)))
    else:
        weights = columns.map(lambda x: weights.loc[x].values).map(lambda x: x / np.sum(x))

    return pd.DataFrame({k: ds[v].values @ weights[k] for k, v in columns.items()}, index=ds.index).rename_axis(name, axis=1)


def vdid_sign(row):
    if row['lb'] < 0 and row['ub'] < 0:
        return '-'
    elif row['lb'] > 0 and row['ub'] > 0:
        return '+'
    else:
        return '+/-'


def vdid_ratio(dx, ratio):
    dx = dx.unstack('target')
    for name, (num, denom) in ratio.items():
        dx = dx.assign(**{name: lambda dx: dx[num] / dx[denom]})
    return dx.stack()


def vdid(dx: pd.DataFrame,
         keys: List[str],
         targets: List[str],
         diffs: List[tuple],
         randomize: str = None,
         ci: object = vdid_jackknife(),
         conf=95,
         ratio=None
         ):
    if ratio is None:
        ratio = dict()

    if randomize is None:
        randomize, _ = diffs[-1]

    labels = {k: v for k, v in diffs}
    did = list(labels.keys())

    index = keys + [k for k, _ in diffs] + [v for _, v in diffs]

    dx = dx.reset_index() if dx.index.name is not None else dx
    for e in index:
        if e not in dx.columns:
            assert f"Column {e} is missing in the provided data frame."
    for e in did:
        assert dx[e].dtype == bool, f"All DiD columns need to be of type bool, but {e} is not."

    dx = dx.groupby(index, as_index=False)[targets].sum()
    dx = pd.melt(dx, id_vars=index, var_name='target', value_name='value').set_index(index + ['target'])['value']

    # grouping along the difference list that was provided
    davg = dx
    counts = dict()

    davg, counts = vdid_avg(davg, [(k, v) for (k, v) in diffs if k != randomize], counts=counts)

    units = davg.reset_index().groupby(randomize)[labels[randomize]].unique()
    counts[randomize] = units.map(lambda x: len(x))

    matrix = davg.droplevel(axis='index', level=randomize).unstack(labels[randomize]).fillna(0.0)
    dagg = vdid_ratio(dot_by_columns(matrix, units, randomize).stack(), ratio).unstack(did)

    # calculate the differences from the aggregated data
    dte_avg = vdid_did(dagg).to_frame('te')

    # if confidence intervals should be calculated
    if ci is not None:
        assert randomize in labels, f"Column to randomize over {randomize} is not available in diffs: {labels}"

        # get the confidence interval sampling method
        ci_sample, ci_fit = ci

        # simulate based on the standard error method
        ci_samp_dict = {sample: vdid_ratio(dot_by_columns(matrix, treatment_mod, randomize).stack(), ratio) for sample, treatment_mod in
                        enumerate(ci_sample(units))}

        ci_samp = pd.DataFrame(ci_samp_dict).rename_axis('sample', axis=1).stack()

        # calculate the did and determine the standard error
        ci_did = vdid_did(ci_samp.unstack(did))
        ci_se = ci_did.reset_index(level='sample', drop=True).pipe(lambda dx: group_by_index(dx).apply(ci_fit))

        # calculate the confidence intervals
        dte_avg = vdid_ci(dte_avg.join(ci_se.to_frame('se')), conf)

    # cumulative treatment effect
    scale = np.prod([count[True] for _, count in counts.items()])
    dte_cum = vdid_ci(dte_avg[['te', 'se']] * scale, conf)

    # percentage treatment effect
    dtreat = vdid_take_by_column(dagg, True)
    dcf = (dte_avg['te'] - dtreat).abs()
    dte_pct = vdid_ci(dte_avg[['te', 'se']].div(dcf.abs() / 100, axis='index'), conf)

    # data frame with all effects in once (avg, pct, cum)
    summary = pd.concat([dte_avg.assign(mode='avg'), dte_pct.assign(mode='pct'), dte_cum.assign(mode='cum')]).sort_index()

    return dict(avg=dte_avg, cum=dte_cum, pct=dte_pct, summary=summary, counts=counts, agg=dagg, scale=scale, cf=dcf)


def vdid_panel(dx, keys, targets, time, unit, **kwargs):
    dte = vdid(dx, keys, targets, [('post', time), ('treatment', unit)], **kwargs)

    dte['avg_agg'] = (dte['agg']
                      .rename(columns={False: 'pre', True: 'post'}, level=0)
                      .rename(columns={False: 'contr', True: 'treat'}, level=1)
                      .pipe(lambda dx: set_to_obj(dx, 'columns', [f'{k}_{v}' for k, v in dx.columns]))
                      .assign(delta_contr=lambda dx: dx['post_contr'] - dx['pre_contr'])
                      .assign(delta_treat=lambda dx: dx['post_treat'] - dx['pre_treat'])
                      .assign(did=lambda dx: dx['delta_treat'] - dx['delta_contr'])
                      )

    dte['cum_agg'] = dte['avg_agg'] * dte['scale']
    dte['pct_agg'] = dte['avg_agg'].div(dte['cf'].abs() / 100, axis='index')

    return dte


def vdid_prepost(dx, keys, targets, time, **kwargs):
    dte = vdid(dx, keys, targets, [('post', time)], **kwargs)

    dte['avg_agg'] = dte['agg'].rename(columns={False: 'pre', True: 'post'}, level=0)
    dte['cum_agg'] = dte['avg_agg'] * dte['scale']
    dte['pct_agg'] = dte['avg_agg'].div(dte['cf'].abs() / 100, axis='index')

    return dte
