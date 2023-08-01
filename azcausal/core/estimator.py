import logging

from numpy.random import RandomState

from azcausal.core.parallelize import Serial


class Estimator(object):

    def __init__(self, verbose=False, name=None, random_state=RandomState(42)) -> None:
        super().__init__()

        if name is None:
            name = self.__class__.__name__.lower()

        self.name = name
        self.verbose = verbose
        self.logger = logging.getLogger()
        self.random_state = random_state

    def fit(self, panel):
        pass

    def error(self, result, method, parallelize=Serial()):
        return method.run(result, parallelize=parallelize)

    def run(self, panel, error=None, parallelize=Serial()):
        result = self.fit(panel)

        if error is not None:
            self.error(result, error, parallelize=parallelize)

        return result

