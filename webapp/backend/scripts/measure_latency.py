"""Assignment 1 target: a 500-word passage is analysed in under 2 seconds.

Measures the full HTTP round trip (validation, cleaning, spaCy, model, JSON)
against a running server. The passages are exactly 500 and 1,000 words under the
app's own counting rule (count_input_words). This times the API only; the
browser adds rendering time on top, so also watch one check in the browser.

    python scripts/measure_latency.py --runs 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.schemas import count_input_words  # noqa: E402

SAMPLES = json.loads((Path(__file__).parent / "data" / "samples.json").read_text(encoding="utf-8"))


def passage(words: int) -> str:
    """Exactly `words` words under the app's rule (punctuation-only tokens don't count)."""
    tokens, text = " ".join(SAMPLES["ai"] + SAMPLES["human"]).split() * 3, []
    for token in tokens:
        text.append(token)
        if count_input_words(" ".join(text)) == words:
            return " ".join(text)
    raise ValueError("not enough sample text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as client:
        for words in (500, 1000):
            text = passage(words)
            assert count_input_words(text) == words
            client.post("/api/detect", json={"text": text})  # warm-up
            times = []
            for _ in range(args.runs):
                start = time.perf_counter()
                r = client.post("/api/detect", json={"text": text})
                times.append(time.perf_counter() - start)
                r.raise_for_status()
            times.sort()
            p95 = times[max(0, int(len(times) * 0.95) - 1)]
            print(
                f"{words:>5} words, {r.json()['sentence_count']} sentences, {args.runs} runs: "
                f"median {statistics.median(times) * 1000:.0f} ms, 95th percentile {p95 * 1000:.0f} ms, "
                f"max {max(times) * 1000:.0f} ms  (target < 2000 ms)"
            )


if __name__ == "__main__":
    main()
