"""Detection-evasion attack (section 2.8): measure it before and after the defence.

The attacker's goal: make AI-written text score as human without changing what
a reader sees. Three perturbations are applied to AI essays the detector
currently catches, at several rates:
    homoglyph   swap Latin letters for identical-looking Cyrillic ones
    zero_width  insert invisible zero-width spaces inside words
    fullwidth   swap letters for fullwidth forms (ａ instead of a)

Evasion success rate = share of caught essays (score > 0.5) whose perturbed
version scores <= 0.5, or gets no score at all (e.g. fully fullwidth text has
no ASCII words, so every sentence falls under the 5-word rule - the attacker
still avoids an AI verdict). It is measured twice on the same perturbed texts:
with NORMALISE_INPUT off (no defence) and on (defence).

Runs the model in-process, so the server does not need to be running.
From the repo root:
    python security/attacks/evasion_attack.py --docs 300
"""
import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend'))

from app.ml.normalise import CONFUSABLES          # noqa: E402
from app.ml.service import DetectorService, NoScorableSentences   # noqa: E402

# Latin letter -> Cyrillic/Greek look-alike: the inverse of the defence's table.
HOMOGLYPHS = {latin: fake for fake, latin in CONFUSABLES.items() if latin.islower()}
ZERO_WIDTH = '​'
THRESHOLD = 0.5


def homoglyph(text: str, rate: float, rng: random.Random) -> str:
    return ''.join(HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < rate else c for c in text)


def zero_width(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for prev, c in zip(' ' + text, text):
        if prev.isalpha() and c.isalpha() and rng.random() < rate:
            out.append(ZERO_WIDTH)
        out.append(c)
    return ''.join(out)


def fullwidth(text: str, rate: float, rng: random.Random) -> str:
    # Fullwidth forms sit at a fixed offset (0xFEE0) from ASCII '!'..'~'.
    return ''.join(chr(ord(c) + 0xFEE0) if c.isascii() and c.isalpha() and rng.random() < rate
                   else c for c in text)


ATTACKS = {'homoglyph': homoglyph, 'zero_width': zero_width, 'fullwidth': fullwidth}
RATES = [0.1, 0.25, 0.5, 1.0]


def load_ai_essays(n: int, seed: int) -> list[str]:
    path = REPO / 'data' / 'processed' / 'drcat_test.csv'
    df = pd.read_csv(path)
    df = df[df.label == 1]
    df = df[df.text.str.split().str.len() <= 1000]     # the app's word limit
    return df.sample(n=min(n, len(df)), random_state=seed).text.tolist()


def score(service: DetectorService, text: str) -> float | None:
    try:
        return service.analyse(text).document['score']
    except NoScorableSentences:
        return None


def main(args):
    rng = random.Random(args.seed)
    model_dir = REPO / 'backend' / 'model'
    undefended = DetectorService(model_dir, normalise_input=False, torch_threads=4)
    defended = DetectorService(model_dir, normalise_input=True, torch_threads=4)

    essays = load_ai_essays(args.docs, args.seed)
    caught = [t for t in essays if score(undefended, t) > THRESHOLD]
    print(f'{len(caught)} of {len(essays)} AI essays are caught before any attack')

    rows = []
    start = time.perf_counter()
    for attack_name, attack in ATTACKS.items():
        for rate in RATES:
            for i, essay in enumerate(caught):
                perturbed = attack(essay, rate, rng)
                rows.append({
                    'attack': attack_name, 'rate': rate, 'essay': i,
                    'score_no_defence': score(undefended, perturbed),
                    'score_defence': score(defended, perturbed),
                })
            print(f'  {attack_name:<10} rate {rate:<4} done ({time.perf_counter() - start:.0f}s)')

    df = pd.DataFrame(rows)
    out_dir = REPO / 'security' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'evasion_scores.csv', index=False)

    # NaN (no score) compares False with >, so it counts as evaded.
    summary = (df.assign(evaded_no_defence=~(df.score_no_defence > THRESHOLD),
                         evaded_defence=~(df.score_defence > THRESHOLD))
                 .groupby(['attack', 'rate'])
                 .agg(essays=('essay', 'count'),
                      evasion_no_defence=('evaded_no_defence', 'mean'),
                      evasion_defence=('evaded_defence', 'mean'),
                      mean_score_no_defence=('score_no_defence', 'mean'),
                      mean_score_defence=('score_defence', 'mean'))
                 .reset_index())
    summary.to_csv(out_dir / 'evasion_summary.csv', index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # Cost of the defence: extra time per request spent normalising.
    clean_times = {}
    for name, service in (('no_defence', undefended), ('defence', defended)):
        t0 = time.perf_counter()
        for essay in caught[:50]:
            service.analyse(essay)
        clean_times[name] = (time.perf_counter() - t0) / min(50, len(caught)) * 1000
    print(f"\nMean latency on clean essays: {clean_times['no_defence']:.1f} ms without, "
          f"{clean_times['defence']:.1f} ms with normalisation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--docs', type=int, default=300, help='AI essays to sample from the DRCAT test split')
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
