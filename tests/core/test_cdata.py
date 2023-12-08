from azcausal.data import CaliforniaProp99

cdf = CaliforniaProp99().cdf()

panel = CaliforniaProp99().panel()

def test_units():
    assert panel.n_units() == cdf.n_units()
    assert panel.n_units(treat=True) == cdf.n_units(treat=True)
    assert panel.n_units(contr=True) == cdf.n_units(contr=True)


def test_times():
    assert panel.n_times() == cdf.n_times()
    assert panel.n_times(pre=True) == cdf.n_times(pre=True)
    assert panel.n_times(post=True) == cdf.n_times(post=True)


def test_filter():
    assert panel.filter(treat=True).n_units() == cdf.filter(treat=True).n_units()
    assert panel.filter(contr=True).n_units() == cdf.filter(contr=True).n_units()

    assert panel.filter(pre=True).n_times() == cdf.filter(pre=True).n_times()
    assert panel.filter(post=True).n_times() == cdf.filter(post=True).n_times()




