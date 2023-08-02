import os
from os.path import join, dirname
from pathlib import Path

import pytest


def filter_by_exclude(files, exclude=[]):
    return [f for f in files if not any([f.endswith(s) for s in exclude])]


def files_from_folder(folder, regex='**/*.py', skip=[]):
    files = [join(folder, fname) for fname in Path(folder).glob(regex) if not os.path.basename(fname).startswith('aws_')]
    # files = [join(folder, fname) for fname in Path(folder).glob(regex)]

    return filter_by_exclude(files, exclude=skip)


def run_file(f):
    fname = os.path.basename(f)

    print("RUNNING:", fname)

    with open(f) as f:
        s = f.read()

        no_plots = "import matplotlib\n" \
                   "matplotlib.use('Agg')\n" \
                   "import matplotlib.pyplot as plt\n" \
                   "__name__ = '__main__'\n"

        s = no_plots + s + "\nplt.close()\n"

        exec(s, globals())

ROOT = dirname(dirname(os.path.realpath(__file__)))


EXAMPLES = join(ROOT, "examples")

SKIP = []

@pytest.mark.long
@pytest.mark.parametrize('example', files_from_folder(EXAMPLES, skip=SKIP))
def test_examples(example):
    run_file(example)
    assert True
