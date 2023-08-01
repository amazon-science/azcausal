from azcausal.cloud.serialization import Serialization


class Server:

    def handle(self, context):

        status = 'ok'
        error = None

        try:
            callable = Serialization.backward(context["payload"].encode('utf-8'))
            obj = callable()
            payload = Serialization.forward(obj).decode('utf-8')

        except Exception as ex:
            status = 'error'
            error = str(ex)
            payload = Serialization.forward(ex).decode('utf-8')

        return dict(status=status, error=error, payload=payload)


def flask_server():
    import json
    from flask import Flask, request, Response

    app = Flask(__name__)

    server = Server()

    @app.route("/", methods=['POST'])
    def run():
        result = server.handle(request.json)
        response = json.dumps(result)
        return Response(response=response, status=200, mimetype="application/json", direct_passthrough=True)

    return app


if __name__ == "__main__":
    flask_server().run(host='0.0.0.0')
