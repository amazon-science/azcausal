import uuid
from os.path import join

import dill
import s3fs


class Job:

    def __init__(self, path=None, prefix=None) -> None:
        super().__init__()

        if path is None:
            path = join(prefix, str(uuid.uuid4()))

        self.path = path

    @property
    def path_to_result(self):
        return self.path + ".result"

    def s3(self):
        return s3fs.S3FileSystem(anon=False)

    def upload(self, obj, path=None):

        if path is None:
            path = self.path

        with self.s3().open(path, 'wb') as f:
            payload = dill.dumps(obj)
            f.write(payload)

    def download(self, path=None):
        if path is None:
            path = self.path

        with self.s3().open(path, 'rb') as f:
            return dill.loads(f.read())

    def delete(self, path=None):
        if path is None:
            path = self.path

        self.s3().delete(path)

    # this function call happens in the AWS Lambda function call
    def __call__(self):
        obj = self.download()
        res = self.run(obj)
        self.upload(res, path=self.path_to_result)

    def result(self):
        return self.download(path=self.path_to_result)

    def run(self, obj):
        pass
