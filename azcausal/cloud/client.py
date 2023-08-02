import json

import requests

from azcausal.cloud.serialization import Serialization


class Endpoint:

    def send(self, content):
        pass


class AWSLambda(Endpoint):

    def __init__(self, client, arn) -> None:
        super().__init__()
        self.client = client
        self.arn = arn

    def send(self, content):
        resp = self.client.invoke(FunctionName=self.arn, Payload=json.dumps(content))
        return json.loads(resp['Payload'].read())


class REST(Endpoint):

    def __init__(self, url, auth=None) -> None:
        super().__init__()
        self.url = url
        self.auth = auth

    def send(self, content):
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(self.url, json=content, auth=self.auth, headers=headers)

        if resp.ok:
            return resp.json()
        else:
            raise Exception(f"Error while send message to endpoint: {resp.text}")


class Client(object):

    def __init__(self, endpoint) -> None:
        super().__init__()
        self.endpoint = endpoint

    def send(self, function, **kwargs):

        # prepare the arguments for the lambda function
        payload = Serialization().forward(function).decode('utf-8')
        content = dict(payload=payload, **kwargs)

        resp = self.endpoint.send(content)

        result = resp.get('payload')
        if result is not None:
            result = Serialization.backward(result.encode('utf-8'))

        if resp.get('status') == "error":
            if isinstance(result, Exception):
                raise Exception("Error while executing callable remotely.") from result
            else:
                raise Exception("Error while executing callable remotely.")

        return result

