import boto3
from arnparse import arnparse
from botocore.config import Config

from azcausal.cloud.client import AWSLambda, Client


class Function:

    def __call__(self):
        from azcausal.core.panel import Panel
        from azcausal.data import CaliforniaProp99
        from azcausal.estimators.panel.sdid import SDID
        from azcausal.util import to_matrices

        # load an example data set with the columns Year, State, PacksPerCapita, treated.
        df = CaliforniaProp99(cache=False).load()

        # convert to matrices where the index represents each Year (time) and each column a state (unit)
        outcome, intervention = to_matrices(df, "Year", "State", "PacksPerCapita", "treated")

        # create a panel object to access observations conveniently
        panel = Panel(outcome, intervention)

        # initialize an estimator object, here synthetic difference in difference (sdid)
        estimator = SDID()

        # run the estimator
        result = estimator.fit(panel)

        return result


if __name__ == "__main__":
    arn = "arn:aws:lambda:us-east-1:112353327285:function:azcausal-lambda-run:$LATEST"

    config = Config(connect_timeout=300, read_timeout=300)
    lambda_client = boto3.client('lambda', region_name=arnparse(arn).region, config=config)

    endpoint = AWSLambda(lambda_client, arn)
    client = Client(endpoint)

    function = Function()
    result = client.send(function)

    print(result.summary())
