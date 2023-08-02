import glob
import itertools
from os.path import dirname, join
from pathlib import Path

import setuptools
from setuptools import find_packages

from azcausal.version import __version__


# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------

def load_requirements():
    res = {}
    for path in glob.glob(join(dirname(__file__), 'requirements', '*.txt')):
        with open(path) as f:
            res[Path(path).stem] = f.read().splitlines()

    res['full'] = list(itertools.chain(*res.values()))
    return res


extras_require = load_requirements()

__name__ = "azcausal"
__author__ = "Julian Blank"
__url__ = "https://github.com/amazon-science/azcausal"
data = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>=3.7',
    author_email="blankjul@amazon.com",
    description="Casual Inference",
    license='Apache License 2.0',
    keywords="causality, inference",
    packages=find_packages(include=['azcausal', 'azcausal.*']),
    install_requires=extras_require['core'],
    extras_require=extras_require,
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)


# ---------------------------------------------------------------------------------------------------------
# OTHER METADATA
# ---------------------------------------------------------------------------------------------------------


# update the README.rst to be part of setup
def readme():
    with open('README.rst') as f:
        return f.read()


data['long_description'] = readme()
data['long_description_content_type'] = 'text/x-rst'

setuptools.setup(**data)
