import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arnparse import arnparse
from botocore.config import Config

from azcausal.cloud.client import AWSLambda, Client
from azcausal.cloud.job import Job
from azcausal.core.parallelize import Joblib
from azcausal.core.power import Power
from azcausal.core.synth import SyntheticEffect
from azcausal.data import CaliforniaProp99

__ARN__ = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"
__S3__ = "s3://azcausal-aws-lambda"

from azcausal.estimators.panel.sdid import SDID

config = Config(connect_timeout=300, read_timeout=300)
lambda_client = boto3.client('lambda', region_name=arnparse(__ARN__).region, config=config)


class MyJob(Job):

    def run(self, panel):
        from azcausal.estimators.panel.sdid import SDID
        from azcausal.core.error import JackKnife
        from azcausal.core.result import Result

        estimator = SDID()
        result = estimator.fit(panel)
        estimator.error(result, JackKnife())

        return Result(result.effects)


def f_estimate_aws(panel):
    endpoint = AWSLambda(lambda_client, __ARN__)
    client = Client(endpoint)

    job = MyJob(prefix=__S3__)
    job.upload(panel)
    client.send(job)

    return job.result()


if __name__ == "__main__":

    f_estimate = f_estimate_aws

    # the number of samples for the power (please increase for higher accuracy)
    n_samples = 11

    parallelize = Joblib(n_samples)

    panel = CaliforniaProp99().panel()
    outcome = panel.outcome.loc[:, ~panel.w]

    results = []

    for att in np.linspace(0.0, 0.30, 7):
        treatment = np.zeros_like(outcome.values)
        treatment[-10:, :2] = -att if att != 0.0 else np.nan

        # create synthetic panels where the last 8 time s
        synth_effect = SyntheticEffect(outcome, treatment, mode='perc')

        # run the power analysis
        pw = Power(f_estimate, synth_effect, n_samples, conf=90).run()

        print(f"Percentage Treatment Effect {att:.3f} | (-): {pw['power']['-']:.3%}")

        results.append(dict(att=att, **pw["power"]))

    df = pd.DataFrame(results)

    print("")
    print(df)

    plt.subplots(1, 1, figsize=(12, 4))
    plt.title("Power Analysis")
    plt.plot(100 * df['att'], df['-'], "-o", color="blue", label='-')
    plt.plot(100 * df['att'], df['+'], "-o", color="green", label='+')
    plt.axhline(1.0, color="black", alpha=0.15)
    plt.axhline(0.9, color="black", alpha=0.15, linestyle='--')
    plt.axhline(0.0, color="black", alpha=0.15)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("ATT (%)")
    plt.ylabel("Statistical Power")
    plt.legend()
    plt.show()
