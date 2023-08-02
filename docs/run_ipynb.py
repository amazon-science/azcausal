import glob
from os.path import dirname

import papermill as pm

for nb in glob.iglob(f'{dirname(__file__)}/**/*.ipynb', recursive=True):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb
    )

    print()
    print(nb)

