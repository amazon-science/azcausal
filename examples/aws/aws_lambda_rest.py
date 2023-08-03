import boto3
from requests_auth_aws_sigv4 import AWSSigV4

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

    endpoint = REST(url, auth=AWSSigV4('lambda', region='us-east-1'))
    client = Client(endpoint)

    function = Function()
    result = client.send(function)
    print(result)
