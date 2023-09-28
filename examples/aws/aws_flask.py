import threading
from time import sleep

from azcausal.remote.client import REST, Client
from azcausal.remote.server import flask_server


class Function:

    def __call__(self):
        return "Flask: Hello World!"


if __name__ == "__main__":
    server = threading.Thread(target=lambda: flask_server().run(host='0.0.0.0'), daemon=True)
    server.start()
    sleep(1)

    endpoint = REST("http://localhost:5000")
    client = Client(endpoint)
    function = Function()
    result = client.send(function)

    print(result)
