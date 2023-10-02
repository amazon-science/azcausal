import glob
from os.path import dirname, join

import papermill as pm
import pytest

root = join(dirname(dirname(__file__)), 'docs', 'source')
IPYNBS = glob.iglob(f"{root}/**/*.ipynb", recursive=True)


@pytest.mark.parametrize("nb", IPYNBS)
def test_ipynb(nb):
    print()
    print(nb)

    pm.execute_notebook(
        input_path=nb,
        output_path=nb
    )

    assert True






