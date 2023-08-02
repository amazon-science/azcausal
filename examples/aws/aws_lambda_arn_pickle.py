import uuid
from os.path import join

import boto3
import s3fs
from arnparse import arnparse
from botocore.config import Config

from azcausal.cloud.client import AWSLambda, Client
from azcausal.cloud.serialization import Serialization
from azcausal.data import CaliforniaProp99


class Function:

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path

    def upload(self, panel):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(self.path, 'wb') as f:
            payload = Serialization().forward(panel, as_str=False)
            f.write(payload)

    def __call__(self):
        import s3fs
        from azcausal.cloud.serialization import Serialization
        from azcausal.estimators.panel.sdid import SDID

        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(self.path, 'rb') as f:
            panel = Serialization().backward(f.read(), from_str=False)

        return SDID().fit(panel)


if __name__ == "__main__":
    arn = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"
    path = join("s3://azcausal-aws-lambda", str(uuid.uuid4()))

    panel = CaliforniaProp99().panel()

    function = Function(path)
    function.upload(panel)

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name=arnparse(arn).region, config=config)

    endpoint = AWSLambda(lambda_client, arn)
    client = Client(endpoint)

    result = client.send(function)

    print(result.summary())
