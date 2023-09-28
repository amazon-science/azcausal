import json
from azcausal.remote.server import Server


def lambda_handler(event, _):

    if "body" in event:
        event = json.loads(event["body"])

        result = Server().handle(event)

        return {
            "isBase64Encoded": False,
            "statusCode": 200,
            "body": result,
            "headers": {
                "content-type": "application/json"
            }
        }

    else:
        return Server().handle(event)


if __name__ == "__main__":
    event = {'http': False,
             'payload': 'gASVrgEAAAAAAACMCmRpbGwuX2RpbGyUjAxfY3JlYXRlX3R5cGWUk5QoaACMCl9sb2FkX3R5cGWUk5SMBHR5cGWUhZRSlIwIRnVuY3Rpb26UaASMBm9iamVjdJSFlFKUhZR9lCiMCl9fbW9kdWxlX1-UjAhfX21haW5fX5SMCF9fY2FsbF9flGgAjBBfY3JlYXRlX2Z1bmN0aW9ulJOUKGgAjAxfY3JlYXRlX2NvZGWUk5QoQwIAAZRLAUsASwBLAUsBS0NDBGQBUwCUTowMSGVsbG8gV29ybGQhlIaUKYwEc2VsZpSFlIw7L1VzZXJzL2JsYW5ranVsL3dvcmtzcGFjZS9hemNhdXNhbC9hemNhdXNhbC9jbG91ZC9jbGllbnQucHmUaBBLTEMCBAGUKSl0lFKUY19fYnVpbHRpbl9fCl9fbWFpbl9fCmgQTk50lFKUfZR9lCiMD19fYW5ub3RhdGlvbnNfX5R9lIwMX19xdWFsbmFtZV9flIwRRnVuY3Rpb24uX19jYWxsX1-UdYaUYowHX19kb2NfX5ROjA1fX3Nsb3RuYW1lc19flF2UdXSUUpQpgZQu'}

    # event = {'http': False,
    #          "payload": "gASVggIAAAAAAACMCmRpbGwuX2RpbGyUjBBfY3JlYXRlX2Z1bmN0aW9ulJOUKGgAjAxfY3JlYXRlX2NvZGWUk5QoQxQAAQwBDAEMAQwDCgMUAwoDBgMKApRLAEsASwBLCksGS0NDbGQBZAJsAG0BfQABAGQBZANsAm0DfQEBAGQBZARsBG0FfQIBAGQBZAVsBm0HfQMBAHwBgwCgCKEAfQR8A3wEZAZkB2QIZAmDBVwCfQV9BnwAfAV8BoMCfQd8AoMAfQh8CKAJfAehAX0JfAlTAJQoTksAjAVQYW5lbJSFlIwQQ2FsaWZvcm5pYVByb3A5OZSFlIwEU0RJRJSFlIwLdG9fbWF0cmljZXOUhZSMBFllYXKUjAVTdGF0ZZSMDlBhY2tzUGVyQ2FwaXRhlIwHdHJlYXRlZJR0lCiME2F6Y2F1c2FsLmNvcmUucGFuZWyUaAeMDWF6Y2F1c2FsLmRhdGGUaAmMHmF6Y2F1c2FsLmVzdGltYXRvcnMucGFuZWwuc2RpZJRoC4wNYXpjYXVzYWwudXRpbJRoDYwEbG9hZJSMA2ZpdJR0lChoB2gJaAtoDYwCZGaUjAdvdXRjb21llIwMaW50ZXJ2ZW50aW9ulIwFcGFuZWyUjAllc3RpbWF0b3KUjAZyZXN1bHSUdJSMOy9Vc2Vycy9ibGFua2p1bC93b3Jrc3BhY2UvYXpjYXVzYWwvYXpjYXVzYWwvY2xvdWQvY2xpZW50LnB5lIwBZpRLa0MUDAEMAQwBDAEKAxQDCgMGAwoDBAKUKSl0lFKUY19fYnVpbHRpbl9fCl9fbWFpbl9fCmgjTk50lFKUfZR9lIwPX19hbm5vdGF0aW9uc19flH2Uc4aUYi4="}
    #

    print(json.dumps(event))
    print(lambda_handler(event, None))
