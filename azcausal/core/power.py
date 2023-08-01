from azcausal.core.parallelize import Serial, Parallelize
from azcausal.core.result import Result


class Power:

    def __init__(self, f_estimate, synth_effect, n_samples, conf=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.f_estimate = f_estimate
        self.synth_effect = synth_effect
        self.n_samples = n_samples
        self.conf = conf

    def run(self, parallelize: Parallelize = Serial()):
        f_effects = self.synth_effect.generator(self.n_samples)
        f_panels = (x["panel"] for x in f_effects)

        runs = parallelize(self.f_estimate, f_panels)
        return Power.fit(runs, self.conf)

    @staticmethod
    def fit(runs, conf=95):
        count = {'+': 0, '-': 0, '+/-': 0}
        for run in runs:
            sign = run.effect.sign(conf=conf, no_effect='+/-')
            count[sign] += 1

        power = {k: v / len(runs) for k, v in count.items()}

        return Result({'power': power, 'count': count}, data=dict(runs=runs))
