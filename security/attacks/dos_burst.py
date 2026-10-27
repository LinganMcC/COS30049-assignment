"""Denial-of-service demo: expensive requests and bursts, with and without the defence.

Two experiments against a running back end:

1. expensive  - send the most expensive request the limits allow (1,000 words
                in 200 short sentences, so every sentence is scored) while
                timing GET /api/health. Without the defence, health waits for
                the analysis; with it, health answers in milliseconds.
2. burst      - send --burst analyses at once. Without the defence every one
                queues and latency climbs; with it, the excess gets 503 at once.

Run the back end twice to compare:
    DOS_PROTECTION=false uvicorn app.main:app --port 8000     (vulnerable)
    uvicorn app.main:app --port 8000                          (defended)
Then, from the repo root:
    python security/attacks/dos_burst.py --burst 30
"""
import argparse
import asyncio
import statistics
import time
from collections import Counter

import httpx

SENTENCE = 'The committee carefully reviewed every single proposal. '   # 7 words
EXPENSIVE_TEXT = (SENTENCE * 142).strip()                              # ~994 words, 142 sentences


async def timed(client: httpx.AsyncClient, method: str, path: str, **kwargs):
    start = time.perf_counter()
    response = await client.request(method, path, **kwargs)
    return response.status_code, (time.perf_counter() - start) * 1000


async def probe_health(client: httpx.AsyncClient, stop: asyncio.Event, samples: list):
    while not stop.is_set():
        samples.append((await timed(client, 'GET', '/api/health'))[1])
        await asyncio.sleep(0.05)


async def expensive(client: httpx.AsyncClient):
    print('\n1. One expensive request while probing /api/health')
    stop, samples = asyncio.Event(), []
    probe = asyncio.create_task(probe_health(client, stop, samples))
    await asyncio.sleep(0.2)
    status, ms = await timed(client, 'POST', '/api/analyse', json={'text': EXPENSIVE_TEXT})
    stop.set()
    await probe
    print(f'   analysis: HTTP {status} in {ms:.0f} ms')
    print(f'   /api/health during it: {len(samples)} probes, median {statistics.median(samples):.0f} ms, '
          f'worst {max(samples):.0f} ms')


async def burst(client: httpx.AsyncClient, n: int):
    print(f'\n2. Burst of {n} expensive requests at once')
    stop, samples = asyncio.Event(), []
    probe = asyncio.create_task(probe_health(client, stop, samples))
    start = time.perf_counter()
    results = await asyncio.gather(*[
        timed(client, 'POST', '/api/analyse', json={'text': EXPENSIVE_TEXT}) for _ in range(n)])
    total = time.perf_counter() - start
    stop.set()
    await probe

    counts = Counter(status for status, _ in results)
    for status in sorted(counts):
        times = [ms for s, ms in results if s == status]
        print(f'   HTTP {status}: {counts[status]:>3} requests, median {statistics.median(times):.0f} ms, '
              f'worst {max(times):.0f} ms')
    print(f'   all answered in {total:.1f} s')
    print(f'   /api/health during it: median {statistics.median(samples):.0f} ms, worst {max(samples):.0f} ms')


async def main(args):
    limits = httpx.Limits(max_connections=args.burst + 5)
    async with httpx.AsyncClient(base_url=args.url, timeout=120, limits=limits) as client:
        await expensive(client)
        await burst(client, args.burst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--burst', type=int, default=30)
    asyncio.run(main(parser.parse_args()))
