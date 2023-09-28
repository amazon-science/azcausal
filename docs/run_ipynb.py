import glob
from os.path import dirname, join

import papermill as pm

for nb in glob.iglob(f"{join(dirname(__file__), 'source')}/**/*.ipynb", recursive=True):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb
    )

    print()
    print(nb)

