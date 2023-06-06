import numpy as np

from azcausal.core.estimator import Estimator


def did(pre_contr, post_contr, pre_treat, post_treat):

    # difference of control
    delta_contr = (post_contr - pre_contr)

    # difference of treatment
    delta_treat = (post_treat - pre_treat)

    # finally the difference in difference
    return delta_treat - delta_contr


class DID(Estimator):

    def fit(self, pnl, lambd=None):

        # get the outcome values pre and post and contr and treat in a block each
        Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = pnl.Y_as_block(trim=True)

        # lambda can be set to define the weights in pre - otherwise it will be simply uniform
        m = pnl.n_pre
        if lambd is None:
            lambd = np.full(m, 1 / m)

        assert len(lambd) == m, f"The input weights lambda must be the same length as pre experiment: {m}"

        # difference for control regions
        pre_contr = Y_pre_contr.mean(axis=0) @ lambd
        post_contr = Y_post_contr.mean(axis=0).mean()

        # difference in treatment
        pre_treat = Y_pre_treat.mean(axis=0) @ lambd
        post_treat = Y_post_treat.mean(axis=0).mean()

        # finally the difference in difference
        att = did(pre_contr, post_contr, pre_treat, post_treat)

        return dict(name="did", estimator=self, panel=pnl, att=att,
                    pre_contr=pre_contr, post_contr=post_contr, pre_treat=pre_treat, post_treat=post_treat)
