"""Denial-of-service defence: a bounded gate in front of the model.

The problem: one analysis costs ~10,000x more than sending the request. If
inference ran inside the `async def` endpoint, it would block FastAPI's single
event loop, so ONE slow request would freeze every other user - even
/api/health. And with no cap on concurrent work, a burst of requests would
pile up until the server ran out of CPU and memory.

The defence has three parts:
    1. Inference runs on a dedicated pool of INFERENCE_WORKERS threads, so the
       event loop stays free to answer other requests.
    2. At most workers + INFERENCE_QUEUE analyses are admitted at once. The
       next one is rejected immediately with 503 + Retry-After, instead of
       waiting in an unbounded queue (fail fast, stay responsive).
    3. An analysis that waits or runs longer than INFERENCE_TIMEOUT_SECONDS
       returns 503 to the client. Its slot is only freed when the thread
       really finishes, so timed-out work can never push us past the cap.

Input limits elsewhere (8,000 characters, 1,000 words, 64 tokens per sentence)
bound the cost of any single request; this module bounds how many run at once.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from app.errors import AppError

T = TypeVar('T')


class InferenceGate:
    def __init__(self, workers: int, queue: int, timeout_seconds: float):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='inference')
        self._capacity = workers + queue
        self._timeout = timeout_seconds
        # Only read and written on the event loop thread, so no lock is needed.
        self._admitted = 0

    @property
    def admitted(self) -> int:
        return self._admitted

    async def run(self, fn: Callable[..., T], *args) -> T:
        if self._admitted >= self._capacity:
            raise AppError(503, 'server_busy',
                           'The detector is busy. Please try again in a few seconds.',
                           headers={'Retry-After': '2'})

        loop = asyncio.get_running_loop()
        self._admitted += 1
        future = self._executor.submit(fn, *args)
        # The callback runs on the worker thread; hand the release back to the loop.
        future.add_done_callback(lambda _: loop.call_soon_threadsafe(self._release))

        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), self._timeout)
        except TimeoutError:
            # A queued job is cancelled here; a running one finishes in the
            # background and frees its slot then.
            raise AppError(503, 'analysis_timeout',
                           'The analysis took too long. Please try a shorter text.',
                           headers={'Retry-After': '5'}) from None

    def _release(self) -> None:
        self._admitted -= 1

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
