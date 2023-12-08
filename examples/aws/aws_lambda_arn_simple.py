import boto3
from botocore.config import Config

from azcausal.remote.client import AWSLambda, Client
from azcausal.util import parse_arn


class Function:

    def __call__(self):
        from azcausal.data import CaliforniaProp99
        from azcausal.estimators.panel.sdid import SDID
        panel = CaliforniaProp99(cache=False).panel()
        estimator = SDID()
        result = estimator.fit(panel)
        return result


if __name__ == "__main__":
    arn = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name=parse_arn(arn)['region'], config=config)

    endpoint = AWSLambda(lambda_client, arn)
    client = Client(endpoint)

    function = Function()
    result = client.send(function)

    print(result.summary())
