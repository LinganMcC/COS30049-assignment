"""Section 2.8: detection evasion, measured before and after the defence.

Attack: change how AI text is ENCODED without changing how it LOOKS, so the
detector calls it human-written.

    zero_width      insert U+200B inside long words               (deterministic)
    homoglyph_N     swap N% of a/c/e/i/o/p/x/y for Cyrillic look-alikes  (random)
    combining       add a combining accent after every 4th letter  (deterministic)
    mixed           zero-width + 50% look-alikes                   (random)
    visible_edits   delete commas and transition words: a real rewrite that
                    normalisation cannot and should not undo       (deterministic)

Random attacks are drawn 5 times per passage (different seeds); deterministic
attacks give the same text every time, so they are run once per passage.
"Success" = the passage scored >= 0.5 before the attack and < 0.5 after it.

Two ways to run it, same inputs and same model:

    python scripts/attacks/evasion_attack.py
        in-process, defence OFF and ON side by side, plus the defence's cost

    python scripts/attacks/evasion_attack.py --api http://127.0.0.1:8000
        through the real POST /api/detect endpoint (validation, cleaning and all).
        Run once with EVASION_DEFENCE_ENABLED=false and once with true.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

MODEL = BACKEND / "model" / "logistic_regression.joblib"
SAMPLES = json.loads((BACKEND / "scripts" / "data" / "samples.json").read_text(encoding="utf-8"))
LOOKALIKE = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "i": "і"}
DRAWS = 5


def zero_width(text, rng):
    return re.sub(r"([A-Za-z]{3})(?=[A-Za-z]{2})", "\\1\u200b", text)


def homoglyph(rate):
    def attack(text, rng):
        return "".join(LOOKALIKE[ch] if ch in LOOKALIKE and rng.random() < rate else ch for ch in text)
    return attack


def combining(text, rng):
    out, letters = [], 0
    for ch in text:
        out.append(ch)
        if ch.isalpha():
            letters += 1
            if letters % 4 == 0:
                out.append("\u0301")
    return "".join(out)


def mixed(text, rng):
    return homoglyph(0.5)(zero_width(text, rng), rng)


def visible_edits(text, rng):
    text = re.sub(r"\b(Furthermore|Moreover|Additionally|However|Consequently|Therefore|Nevertheless|"
                  r"Ultimately|Overall|Notably|Similarly|In conclusion|In summary|Firstly), ", "", text)
    return text.replace(",", "")


# name -> (function, is_random)
ATTACKS = {
    "zero_width": (zero_width, False),
    "homoglyph_10": (homoglyph(0.10), True),
    "homoglyph_30": (homoglyph(0.30), True),
    "homoglyph_100": (homoglyph(1.0), False),  # rate 1.0 replaces every candidate: deterministic
    "combining": (combining, False),
    "mixed": (mixed, True),
    "visible_edits": (visible_edits, False),
}


def variants(text, index, attack, is_random):
    seeds = range(DRAWS) if is_random else [0]
    return [attack(text, random.Random(index * 100 + k)) for k in seeds]


def run(score_fn, label):
    texts = SAMPLES["ai"]
    base = [score_fn(t) for t in texts]
    eligible_passages = sum(b >= 0.5 for b in base)
    print(f"\n[{label}] {len(texts)} AI-written passages, {eligible_passages} scored >= 0.5 before any attack "
          f"(mean {statistics.mean(base):.3f})")
    print(f"{'attack':15s} {'variants':>8s} {'eligible':>8s} {'success':>8s} {'mean':>7s} {'min':>7s}")
    results = {}
    for name, (attack, is_random) in ATTACKS.items():
        scores, success, eligible = [], 0, 0
        for i, (text, before) in enumerate(zip(texts, base)):
            for variant in variants(text, i, attack, is_random):
                after = score_fn(variant)
                scores.append(after)
                if before >= 0.5:
                    eligible += 1
                    success += after < 0.5
        rate = success / eligible if eligible else 0.0
        results[name] = {"variants": len(scores), "eligible": eligible, "success_rate": rate,
                         "mean_score": statistics.mean(scores), "min_score": min(scores)}
        print(f"{name:15s} {len(scores):8d} {eligible:8d} {rate:8.0%} {statistics.mean(scores):7.3f} {min(scores):7.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="base URL of a running API, e.g. http://127.0.0.1:8000")
    args = parser.parse_args()

    if args.api:
        import httpx
        client = httpx.Client(base_url=args.api, timeout=30)
        def score(text):
            r = client.post("/api/detect", json={"text": text})
            r.raise_for_status()
            return r.json()["score"]
        run(score, f"through {args.api}/api/detect (defence as set in the server's .env)")
        return

    from app.ml.detector import Detector
    from app.ml.text_pipeline import normalise_for_evasion

    cfg = MODEL.parent / "frozen_config.json"
    off = Detector.load(MODEL, cfg, evasion_defence=False)
    on = Detector.load(MODEL, cfg, evasion_defence=True)
    summary = {
        "defence_off": run(lambda t: off.analyse(t)["score"], "defence OFF"),
        "defence_on": run(lambda t: on.analyse(t)["score"], "defence ON"),
    }

    print("\nCost of the defence on ordinary (un-attacked) text: score with defence ON minus OFF")
    for group in ("ai", "human", "accented", "non_latin"):
        diffs = [on.analyse(t)["score"] - off.analyse(t)["score"] for t in SAMPLES[group]]
        print(f"  {group:10s} {len(diffs):2d} passages: largest change {max(abs(d) for d in diffs):.4f}")
    text = " ".join(SAMPLES["ai"])
    start = time.perf_counter()
    for _ in range(200):
        normalise_for_evasion(text)
    norm_ms = (time.perf_counter() - start) / 200 * 1000
    start = time.perf_counter()
    for t in SAMPLES["ai"]:
        on.analyse(t)
    full_ms = (time.perf_counter() - start) / len(SAMPLES["ai"]) * 1000
    print(f"  normalisation time: {norm_ms:.2f} ms for {len(text.split())} words (a full analysis takes ~{full_ms:.0f} ms)")

    out = BACKEND / "scripts" / "attacks" / "evasion_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(BACKEND)}")


if __name__ == "__main__":
    main()
