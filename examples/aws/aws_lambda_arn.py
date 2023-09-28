import boto3
from botocore.config import Config

from azcausal.remote.client import AWSLambda, Client


class Function:

    def __call__(self):
        return "AWS Lambda (ARN): Hello World!"


if __name__ == "__main__":
    arn = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name='us-east-1', config=config)

    endpoint = AWSLambda(lambda_client, arn)
    client = Client(endpoint)

    function = Function()
    result = client.send(function)
    print(result)
