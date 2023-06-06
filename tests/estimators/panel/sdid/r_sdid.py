import numpy as np
import rpy2.robjects as ro
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

from azcausal.data import CaliforniaProp99
from tests.estimators.panel.test_sdid import data_generator

base = rpackages.importr('base')
synthdid = rpackages.importr('synthdid')


def pd_to_r(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)

def to_matrix(rdf):
    return robjects.r['data.matrix'](rdf)

def obj_to_dict(obj):
    return {k: obj.slots[k] for k in obj.slots}


def vector_to_dict(obj):
    return dict(zip(obj.names, list(obj)))


def panel_to_matrix(df):
    # pnlects the columns in order:  unit = 1, time = 2, outcome = 3, treatment = 4, treated.last = TRUE
    Y, N0, T0, _ = synthdid.panel_matrices(df)
    return Y, N0, T0


def sdid(Y, N0, T0):
    estm = synthdid.synthdid_estimate(Y, N0, T0)
    tau_hat = float(estm[0])
    return tau_hat, estm


def error(estm, method):
    res = synthdid.vcov_synthdid_estimate(estm, method=method)
    se = float(base.sqrt(res[0])[0])
    return se


def sdid_bootstrap_sample(df, n_samples=100):
    Y, N0, T0 = panel_to_matrix(pd_to_r(df))
    tau_hat, estm = sdid(Y, N0, T0)
    samples = synthdid.bootstrap_sample(estm, n_samples)
    return np.array(samples)


def sdid_placebo_sample(df, n_samples=100):
    r_source = robjects.r['source']

    Y, N0, T0 = panel_to_matrix(pd_to_r(df))
    tau_hat, estm = sdid(Y, N0, T0)
    samples = r_source("custom.r")[0](estm, n_samples)
    return np.array(samples)


def sdid_from_panel(df, **kwargs):
    Y, N0, T0 = panel_to_matrix(pd_to_r(df))
    return sdid_from_matrix(Y, N0, T0, **kwargs)


def sdid_from_matrix(Y, N0, T0):
    treat_units = list(Y.names[0])[int(N0[0]):]
    treat_start = list(Y.names[1])[int(T0[0])]

    tau_hat, estm = sdid(Y, N0, T0)
    se = error(estm, method='jackknife')

    weights = vector_to_dict(estm.slots["weights"])

    lambd = list(weights["lambda"])
    f_min_lambd = min(weights["lambda.vals"])

    omega = list(weights["omega"])
    f_min_omega = min(weights["omega.vals"])

    res = dict(treat_units=treat_units, treat_start=treat_start, tau_hat=tau_hat, se=se,
               omega=omega, lambd=lambd, f_min_lambd=f_min_lambd, f_min_omega=f_min_omega)

    return res


def did_from_panel(df):
    Y, N0, T0 = panel_to_matrix(pd_to_r(df))
    return synthdid.did_estimate(Y, N0, T0)[0]


if __name__ == "__main__":

    df = CaliforniaProp99().load()

    did = did_from_panel(df)
    print(did)

    # n_samples = 5000
    #
    # placebo = sdid_placebo_sample(add_to_treatment(california_prop99, ["Wyoming"], 1989), n_samples=n_samples)
    # with open('placebo.pkl', 'wb') as f:
    #     pickle.dump(placebo, f)

    # boostrap = sdid_bootstrap_sample(add_to_treatment(california_prop99, ["Wyoming"], 1989), n_samples=n_samples)
    # with open('bootstrap.pkl', 'wb') as f:
    #     pickle.dump(boostrap, f)

    res = []
    for df in data_generator():
        estm = sdid_from_panel(df)
        res.append(estm)
    print(res)

    # with open('correct/synthdid.pkl', 'wb') as f:
    #     pickle.dump(res, f)
