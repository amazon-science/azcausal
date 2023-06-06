from abc import abstractmethod
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


class Parallelize(object):

    def __call__(self, func, args):
        return self.run(func, args)

    @abstractmethod
    def run(self, func, args):
        pass


class Serial(Parallelize):

    def __init__(self, progress=False) -> None:
        super().__init__()
        self.progress = progress

    def run(self, func, iterable):
        if self.progress:
            iterable = list(iterable)
            return [func(args) for args in tqdm(iterable, total=len(iterable))]
        else:
            return [func(args) for args in iterable]


class Pool(Parallelize):

    def __init__(self, pool=None, mode="processes", max_workers=None, progress=False) -> None:
        super().__init__()

        if pool is None:

            if max_workers is None:
                max_workers = cpu_count() - 1

            if mode == "processes":
                pool = ProcessPool(max_workers)
            elif mode == "threads":
                pool = ThreadPool(max_workers)

        self.pool = pool
        self.progress = progress

    def run(self, func, iterable):
        if self.progress:
            iterable = list(iterable)
            futures = self.pool.imap(func, iterable)
            return [e for e in tqdm(futures, total=len(iterable))]
        else:
            return self.pool.map(func, iterable)
