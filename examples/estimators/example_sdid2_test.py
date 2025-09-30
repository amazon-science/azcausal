from azcausal.core.panel import CausalPanel
from azcausal.data import CaliforniaProp99
from azcausal.experimental.sdid2 import InstanceFactory, solve, MyPanel, sdid2
from azcausal.util import to_panels
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

if __name__ == "__main__":
    df = CaliforniaProp99().df()

    # create the panel data from the frame and define the causal types
    data = to_panels(df, 'Year', 'State', ['PacksPerCapita', 'treated'])
    ctypes = dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    # initialize the panel
    panel = CausalPanel(data).setup(**ctypes)

    Y = panel["outcome"].values
    YY = np.hstack([Y[:, ~(panel.treat)], Y[:, panel.treat]])
    n_treat = panel.treat.sum()
    n_post = panel.post.sum()

    panel = MyPanel(Y, n_treat, n_post)

    factory = InstanceFactory(panel, 1_000)
    instances = factory.run()

    dx = pd.DataFrame(dict(instance=instances))

    dx['true_instance_att'] = dx['instance'].map(lambda x: x.effect())

    ans = sdid2(Y, n_treat, n_post)


    dx = ans['dx']
    dx['pred_instance_att'] = dx['att']
    dx['true_instance_att'] = dx['instance'].map(lambda x: x.effect())

    print(dx.dropna()['true_instance_att'].describe())
    print(dx.dropna()['pred_instance_att'].describe())

    plt.hist(dx['pred_instance_att'], bins=31)
    plt.show()

    print("sdfsf")
