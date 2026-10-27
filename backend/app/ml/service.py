"""DetectorService - the model as a black box: text in, scores out.

Created once at start-up (see main.py lifespan) and shared by every request.
`analyse` is ordinary blocking code; the caller decides which thread runs it
(see security/concurrency.py), so it never blocks FastAPI's event loop.

Pipeline, identical to transformer/check_my_text.py:
    normalise -> split sentences -> drop < 5 words -> tokenise -> model -> aggregate
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from app.ml.model import TransformerClassifier
from app.ml.normalise import normalise
from app.ml.sentences import MIN_WORDS, count_words, load_spacy, split_sentences
from app.security.integrity import verify_model_files

log = logging.getLogger(__name__)

PAD = '<pad>'
BATCH_SIZE = 32
# Display bands from the A1 design (upper bound inclusive).
BANDS = [(0.25, 'Human'), (0.50, 'Possibly AI'), (0.75, 'Likely AI'), (1.00, 'Very likely AI')]


class NoScorableSentences(ValueError):
    """Every sentence was under MIN_WORDS words, so there is nothing to score."""


@dataclass
class Analysis:
    sentences: list[dict]
    document: dict
    characters_normalised: int
    excluded_sentences: int


def likelihood_band(probability: float) -> str:
    for upper, name in BANDS:
        if probability <= upper:
            return name
    return BANDS[-1][1]


def aggregate_document(probs: np.ndarray, z: float = 1.96, min_variance: float = 0.01) -> dict:
    """Mean sentence score with a ~95% range that widens for short or mixed documents.

    Same formula as train_transformer.aggregate_document: the range is
    mean +/- z * sqrt((variance + min_variance) / n). Few sentences (small n)
    or sentences that disagree (high variance) both widen it.
    """
    n = len(probs)
    mean = float(probs.mean())
    half_width = z * float(np.sqrt((probs.var() + min_variance) / n))
    return {'score': mean, 'low': max(0.0, mean - half_width),
            'high': min(1.0, mean + half_width), 'band': likelihood_band(mean)}


class DetectorService:
    def __init__(self, model_dir: Path, normalise_input: bool, torch_threads: int):
        # Refuses to continue if either file's hash does not match SHA256SUMS.
        verify_model_files(model_dir)

        torch.set_num_threads(torch_threads)
        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        # weights_only=True: torch refuses to unpickle anything except tensors
        # and plain containers, so even a file that passed the hash check
        # cannot run code while loading. Two independent layers of defence.
        saved = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True)
        self.config = saved['config']
        self.model = TransformerClassifier(**self.config)
        self.model.load_state_dict(saved['model_state'])
        self.model.eval()   # dropout off -> identical input gives identical scores (Req 10)

        self.pad_id = self.tokenizer.token_to_id(PAD)
        self.max_len = self.config['max_len']
        self.nlp = load_spacy()
        self.normalise_input = normalise_input
        log.info('Model loaded (%s parameters)', sum(p.numel() for p in self.model.parameters()))

    def analyse(self, text: str) -> Analysis:
        changed = 0
        if self.normalise_input:
            result = normalise(text)
            text, changed = result.text, result.characters_changed
            if changed:
                # Log the count only - never the submitted text (Req 11).
                log.info('Normalised %d suspicious characters', changed)

        sentences = split_sentences(self.nlp, text)
        scorable = [i for i, s in enumerate(sentences) if count_words(s) >= MIN_WORDS]
        if not scorable:
            raise NoScorableSentences(
                f'No sentence has {MIN_WORDS} or more words, so there is nothing to score.')

        probs = self._predict([sentences[i] for i in scorable])
        score_by_index = dict(zip(scorable, probs.tolist()))

        sentence_results = []
        for i, sentence in enumerate(sentences):
            p = score_by_index.get(i)
            sentence_results.append({
                'index': i, 'text': sentence, 'scored': p is not None,
                'score': p, 'band': likelihood_band(p) if p is not None else None,
            })
        document = aggregate_document(probs)
        document['scored_sentences'] = len(scorable)
        return Analysis(sentence_results, document, changed, len(sentences) - len(scorable))

    @torch.inference_mode()
    def _predict(self, sentences: list[str]) -> np.ndarray:
        """P(AI) for each sentence. Sentences longer than max_len tokens are cut,
        exactly as in training - this also caps the model's cost per sentence."""
        ids = [enc.ids[:self.max_len] for enc in self.tokenizer.encode_batch(sentences)]
        out = []
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start + BATCH_SIZE]
            width = max(len(row) for row in batch)
            input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.bool)
            for r, row in enumerate(batch):
                input_ids[r, :len(row)] = torch.tensor(row, dtype=torch.long)
                mask[r, :len(row)] = True
            logits = self.model(input_ids, mask)
            out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out)
