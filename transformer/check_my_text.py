"""Score your own text with the trained sentence-level transformer.

Usage:
    python3 check_my_text.py my_essay.txt       # score a text file
    python3 check_my_text.py                    # paste text, then press Ctrl-D

Does exactly what the website will do:
    split into sentences (spaCy) -> drop sentences under 5 words ->
    score each sentence -> aggregate into a document score + confidence range
"""
import sys

import torch
from tokenizers import Tokenizer

from sentence_splitter import split_sentences
from train_transformer import (TransformerClassifier, encode_sentences, predict_sentences,
                               aggregate_document, likelihood_band, count_words, MIN_WORDS, PAD)

MODEL_DIR = 'results/transformer'


def main():
    # Read the text from a file if one was given, otherwise from what you paste.
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding='utf-8') as f:
            text = f.read()
    else:
        print('Paste your text, then press Ctrl-D on a new line:')
        text = sys.stdin.read()

    # Rebuild the trained model and tokenizer from the files training saved.
    tokenizer = Tokenizer.from_file(f'{MODEL_DIR}/tokenizer.json')
    saved = torch.load(f'{MODEL_DIR}/best_model.pt', map_location='cpu')
    model = TransformerClassifier(**saved['config'])
    model.load_state_dict(saved['model_state'])

    # Same sentence rules as training and the website.
    all_sentences = split_sentences(text)
    sentences = [s for s in all_sentences if count_words(s) >= MIN_WORDS]
    excluded = len(all_sentences) - len(sentences)
    if not sentences:
        print(f'No sentences with {MIN_WORDS}+ words to score.')
        return

    encoded = encode_sentences(sentences, tokenizer, saved['config']['max_len'])
    probs = predict_sentences(model, encoded, tokenizer.token_to_id(PAD), 32, torch.device('cpu'))

    print('\nSENTENCE SCORES')
    for s, p in zip(sentences, probs):
        print(f'  {p:5.0%}  {likelihood_band(p):15s} {s}')

    doc = aggregate_document(probs)
    flagged = sum(p > 0.50 for p in probs)
    print('\nDOCUMENT')
    print(f"  {doc['score']:.0%} estimated AI-generated  ->  {likelihood_band(doc['score'])}")
    print(f"  Confidence range: {doc['low']:.0%} - {doc['high']:.0%}")
    print(f'  Sentences flagged likely/very likely AI: {flagged} of {len(sentences)}')
    print(f'  Highest scoring sentence: {max(probs):.0%}')
    print(f'  Sentences excluded (under {MIN_WORDS} words): {excluded}')


if __name__ == '__main__':
    main()