from azcausal.core.result import Result


class Run(object):

    def __init__(self, estimator, low_memory=False) -> None:
        super().__init__()
        self.estimator = estimator
        self.low_memory = low_memory

    def __call__(self, panel):

        result = self.estimator.fit(panel)

        # only keep the effects of the result (not the data)
        if self.low_memory:
            result = Result(result.effects)

        return result


