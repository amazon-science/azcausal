from collections import Counter

from azcausal.core.parallelize import Serial, Parallelize


class Evaluator:

    def __init__(self, conf=90) -> None:
        super().__init__()
        self.conf = conf

    def __call__(self, synth_effect):
        true = synth_effect['correct']['effect']
        pred = synth_effect['result'].effect
        ci = pred.ci(conf=self.conf)

        perc_true = true.percentage()
        perc_pred = pred.percentage(counter_factual=true.counter_factual)
        perc_ci = perc_pred.ci(conf=self.conf)

        res = {
            **synth_effect['tags'],
            'abs': {
                'att': pred.value,
                'se': pred.se,
                'lb': ci[0],
                'ub': ci[1],
                'true_att': true.value,
            },
            'perc': {
                'att': perc_pred.value,
                'se': perc_pred.se,
                'lb': perc_ci[0],
                'ub': perc_ci[1],
                'true_att': perc_true.value,
            },
            'sign': pred.sign(conf=self.conf),

        }
        return res


class Scenario:

    def __init__(self, f_estimate, synth_effect, f_eval=Evaluator(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.f_estimate = f_estimate
        self.synth_effect = synth_effect
        self.f_eval = f_eval

    def run(self, n_samples, parallelize: Parallelize = Serial()):
        def f(synth_effect):
            synth_effect['result'] = self.f_estimate(synth_effect['panel'])
            return self.f_eval(synth_effect)

        f_effects = self.synth_effect.generator(n_samples)

        return parallelize(f, f_effects, total=n_samples)


def power(results):
    counts = Counter([e['sign'] for e in results])
    return {s: counts[s] / len(results) for s in ['+', '+/-', '-']}

