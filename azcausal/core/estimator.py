from azcausal.core.parallelize import Serial


class Estimator(object):

    def error(self, estm, method, parallelize=Serial()):
        return method.run(estm, "att", parallelize=parallelize)


