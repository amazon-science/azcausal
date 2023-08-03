import uuid
from os.path import join

import boto3
from botocore.config import Config

from azcausal.cloud.client import AWSLambda, Client
from azcausal.data import CaliforniaProp99
from azcausal.util import parse_arn


class Function:

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __call__(self):
        import pandas as pd
        from azcausal.core.panel import Panel
        from azcausal.estimators.panel.sdid import SDID

        data = {k: pd.read_csv(v) for k, v in self.data.items()}
        outcome = data["outcome"].set_index("Year")
        intervention = data["intervention"].set_index("Year")

        # create a panel object to access observations conveniently
        panel = Panel(outcome, intervention)

        # initialize an estimator object, here synthetic difference in difference (sdid)
        estimator = SDID()

        # run the estimator
        result = estimator.fit(panel)

        return result


if __name__ == "__main__":

    folder = join("s3://azcausal-aws-lambda", str(uuid.uuid4()))

    panel = CaliforniaProp99().panel()
    panel.outcome.to_csv(join(folder, 'outcome.csv'))
    panel.intervention.to_csv(join(folder, 'intervention.csv'))

    arn = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name=parse_arn(arn)['region'], config=config)

    endpoint = AWSLambda(lambda_client, arn)
    client = Client(endpoint)

    data = dict(outcome=join(folder, 'outcome.csv'), intervention=join(folder, 'intervention.csv'))
    function = Function(data=data)
    result = client.send(function)

    print(result.summary())
