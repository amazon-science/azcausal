import boto3
from botocore.config import Config

from azcausal.remote.client import AWSLambda, Client
from azcausal.remote.job import Job
from azcausal.data import CaliforniaProp99
from azcausal.util import parse_arn


class MyJob(Job):

    def run(self, panel):
        from azcausal.estimators.panel.did import DID
        from azcausal.core.error import JackKnife
        from azcausal.core.result import Result

        estimator = DID()
        result = estimator.fit(panel)
        estimator.error(result, JackKnife())

        return Result(result.effects)


if __name__ == "__main__":
    __ARN__ = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"
    __S3__ = "s3://azcausal-aws-lambda"

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name=parse_arn(__ARN__).region, config=config)

    panel = CaliforniaProp99().panel()

    endpoint = AWSLambda(lambda_client, __ARN__)
    client = Client(endpoint)

    job = MyJob(prefix=__S3__)
    job.upload(panel)
    client.send(job)
    result = job.result()

    print(result.summary())
