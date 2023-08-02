from urllib.parse import urlparse

import boto3
from aws_requests_auth.aws_auth import AWSRequestsAuth

from azcausal.cloud.client import REST, Client


class Function:

    def __call__(self):
        return "AWS Lambda (REST): Hello World!"


if __name__ == "__main__":

    # Local Docker Image
    # url = "http://localhost:9000/2015-03-31/functions/function/invocations"

    # Remote AWS URL
    url = "https://qfvmzh26gszowwbxnbrorpyj7a0tmdsd.lambda-url.us-east-1.on.aws/"
    # url = "https://ufkauvbiybsxhrwhsoot3hdyda0yxdlo.lambda-url.us-east-1.on.aws/"

    session = boto3.Session()
    credentials = session.get_credentials()

    auth = AWSRequestsAuth(aws_access_key=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key,
                           aws_token=credentials.token,
                           aws_host=urlparse(url).hostname,
                           aws_region='us-east-1',
                           aws_service='lambda')

    endpoint = REST(url, auth=auth)
    client = Client(endpoint)

    function = Function()
    result = client.send(function)
    print(result)
