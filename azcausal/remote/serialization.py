import base64

import dill

METHOD = dill


class Serialization:

    @staticmethod
    def forward(obj, as_str=True):
        ret = METHOD.dumps(obj)
        if as_str:
            ret = base64.urlsafe_b64encode(ret)

        return ret

    @staticmethod
    def backward(s, from_str=True):
        if from_str:
            s = base64.urlsafe_b64decode(s)

        return METHOD.loads(s)
