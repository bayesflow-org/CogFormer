from concurrent.futures import ThreadPoolExecutor


class Prefetcher:
    """Overlap CPU-side batch simulation with GPU training.

    Submits the next batch to a background thread as soon as the current
    one is handed off, so simulation and training run concurrently.
    """

    def __init__(self, sample_fn):
        self._fn = sample_fn
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next = self._pool.submit(self._fn)

    def next(self):
        batch = self._next.result()
        self._next = self._pool.submit(self._fn)
        return batch

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
