
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
# `tokenizers` comes with `transformers` (already in requirements.txt). We only
# use it to TRAIN a fresh vocabulary - nothing pretrained is downloaded.
from tokenizers import ByteLevelBPETokenizer

PAD = '<pad>'    # special token used to fill short sentences so a batch is rectangular
MIN_WORDS = 5    # outline: "sentences under five words are excluded"

# Display bands from the results-screen mockup (upper bound, label).
BANDS = [(0.25, 'Human'), (0.50, 'Possibly AI'), (0.75, 'Likely AI'), (1.00, 'Very likely AI')]


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
# Copied from train_document_models.py so every model reports IDENTICAL
# metrics. Copied rather than imported because importing that file pulls in
# features.py -> sentence_splitter.py, which loads spaCy at import time.

def score(y, probabilities, threshold):
    """Turn probabilities into metrics.

    Why this helps:
        Accuracy, precision, recall and F1 need a yes/no prediction, so they
        depend on the threshold. ROC-AUC and PR-AUC don't - they measure how
        well the model RANKS AI above human overall.

    Example:
        probabilities = [0.9, 0.2, 0.6], threshold = 0.5 -> predictions = [1, 0, 1]
    """
    predicted = (probabilities >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, predicted),
        # precision_ai: of the items we CALLED AI, how many really were AI?
        'precision_ai': precision_score(y, predicted, zero_division=0),
        # recall_ai: of the items that really WERE AI, how many did we catch?
        'recall_ai': recall_score(y, predicted, zero_division=0),
        'f1_ai': f1_score(y, predicted, zero_division=0),
        'roc_auc': roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        'pr_auc': average_precision_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        # Raw confusion-matrix counts: true-neg, false-pos, false-neg, true-pos.
        'tn_fp_fn_tp': confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist(),
    }


def choose_threshold(y, val_probs):
    """Pick the cut-off with the best F1 on VALIDATION only (never on test)."""
    thresholds = np.linspace(0.10, 0.90, 81)
    f1_values = [f1_score(y, val_probs >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1_values))])


def pick_device():
    """cuda = NVIDIA GPU, mps = Apple Silicon GPU, cpu = fallback (slowest)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ----------------------------------------------------------------------
# Step 1: load sentences
# ----------------------------------------------------------------------
def count_words(text):
    """Same word definition as features._words(), so the 5-word rule matches
    what the rest of the project counts as a word."""
    return len(re.findall(r"[A-Za-z']+", str(text)))


def load_sentences(path):
    """Read one sentence CSV from sentence_splitter.py and apply the 5-word rule.

    Why the rule applies in training too:
        The app will never score a sentence under 5 words, so there is no
        point teaching the model on them - and fragments like "Nobody
        mentions wasps." carry too little evidence to label reliably.
    """
    df = pd.read_csv(path).dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df['n_words'] = df['text'].map(count_words)
    kept = df[df['n_words'] >= MIN_WORDS].reset_index(drop=True)
    print(f'{path.name}: kept {len(kept):,} of {len(df):,} sentences '
          f'({len(df) - len(kept):,} under {MIN_WORDS} words excluded)')
    return kept


# ----------------------------------------------------------------------
# Steps 2-3: our own tokenizer, sentences -> token ids
# ----------------------------------------------------------------------
def train_tokenizer(texts, vocab_size, out_dir):
    """Learn a vocabulary of word pieces from the TRAINING sentences.

    BPE (byte-pair encoding) starts from single bytes and repeatedly merges
    the most frequent neighbouring pairs, so common words become one token and
    rare words are built from pieces.

    Why byte-level: any text can be encoded, even words never seen in training
    (important for HC3 and for whatever users paste into the website).
    Why casing/punctuation are kept: they are style signals - the same reason
    data_cleaning.py doesn't lowercase or strip them.
    Leakage rule: only training sentences are used.

    Example (illustrative):
        "Moreover, cars pollute." -> ["More", "over", ",", " cars", " poll", "ute", "."]
                                  -> [812, 1033, 11, 2291, 3870, 544, 13]
    """
    tokenizer = ByteLevelBPETokenizer()   # lowercase=False by default
    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=[PAD])
    tokenizer.save(str(out_dir / 'tokenizer.json'))
    return tokenizer


def encode_sentences(texts, tokenizer, max_len):
    """Tokenise every sentence; cut any sentence longer than max_len tokens.

    64 tokens covers nearly every sentence (roughly 45-50 words), so
    truncation only affects rare run-on sentences.
    """
    encodings = tokenizer.encode_batch([str(t) for t in texts])   # parallel, fast
    return [enc.ids[:max_len] for enc in encodings]


def make_loader(encoded, labels, pad_id, batch_size, shuffle):
    """Group sentences into padded batches.

    A batch must be a rectangle, but sentences have different lengths, so
    short ones are filled with the pad id and an attention mask tells the
    model which positions are real.

    Example batch of 2 sentences:
        input_ids      = [[812, 44, 91], [305, 17, PAD]]
        attention_mask = [[T,   T,  T ], [T,   T,  F  ]]
        labels         = [1, 0]
    """
    items = list(zip(encoded, labels))

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :len(ids)] = True
        return input_ids, attention_mask, torch.tensor([int(y) for _, y in batch])

    # shuffle=True for training (mixes essays/prompts in each batch);
    # shuffle=False for prediction so outputs stay in row order.
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# ----------------------------------------------------------------------
# Step 4: the model (built by us, random starting weights)
# ----------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """A small transformer encoder with a human/AI output layer on top.

    Data flow for one sentence of N tokens:

        token ids (N)
           |  token embedding: each id -> a learned vector of size d_model
           |  + position embedding: a learned vector for position 0, 1, 2 ...
           v     (attention alone has no sense of word order, this adds it)
        N vectors
           |  n_layers x TransformerEncoderLayer:
           |     self-attention -> every token looks at every other token in
           |                       the sentence and mixes in what's relevant
           |     feed-forward   -> a small neural net applied to each token
           v
        N context-aware vectors
           |  mean pooling: average the vectors of the REAL tokens (not padding)
           v
        1 vector summarising the sentence
           |  linear layer
           v
        2 numbers (logits): score for "human", score for "AI"

    Why each sentence is scored ON ITS OWN (no neighbouring sentences):
        The website must highlight mixed text - human sentences next to AI
        ones (see the results-screen mockup). DRCAT essays are entirely human
        or entirely AI, so a model that read the whole document would learn
        "all sentences in a document share a label" and smear one colour over
        a mixed passage. Scoring sentences independently avoids that.
    """

    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, ff_dim, dropout, pad_id):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True,
            # norm_first=True ("pre-norm") trains more stably from scratch.
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        # Dropout randomly zeroes values during training - discourages memorising.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # PyTorch's mask convention is the opposite of ours: True = IGNORE.
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def warmup_then_linear_decay(optimizer, warmup_steps, total_steps):
    """Learning rate rises from ~0 over `warmup_steps`, then falls linearly to 0.

    Random starting weights produce large, noisy gradients; a small learning
    rate at first stops early updates wrecking training. Shrinking it at the
    end lets the model settle.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()   # no gradients needed when only predicting -> faster, less memory
def predict_sentences(model, encoded, pad_id, batch_size, device):
    """Return P(AI) for every sentence, in the same order as `encoded`.

    The model gives 2 logits per sentence, e.g. [0.2, 1.8]; softmax turns them
    into probabilities that sum to 1, e.g. [0.17, 0.83]; we keep the AI one.
    """
    loader = make_loader(encoded, np.zeros(len(encoded)), pad_id, batch_size, shuffle=False)
    model.eval()   # dropout off -> same input always gives the same score (Req 10)
    probs = []
    for input_ids, attention_mask, _ in loader:
        logits = model(input_ids.to(device), attention_mask.to(device))
        probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


# ----------------------------------------------------------------------
# Step 7: sentences -> document score + confidence range (Req 5, Req 6)
# ----------------------------------------------------------------------
def aggregate_document(sentence_probs, z=1.96, min_variance=0.01):
    """Combine one document's sentence scores into a score and confidence range.

    Score = mean of the sentence probabilities.
    Range = mean +/- z * standard error, where
            standard error = sqrt((variance of sentence scores + min_variance) / n)

    Why this matches WBS 2.4 ("the interval widens on short or mixed documents"):
        - FEW sentences (short document) -> divide by a small n -> wider range
        - sentences that DISAGREE (mixed document) -> high variance -> wider range
        - min_variance stops a document whose few sentences happen to agree
          from getting a falsely razor-thin range.
    z = 1.96 corresponds to a ~95% interval. Treat min_variance as a tunable
    design choice - check on validation that the ranges look sensible.

    Examples (4 sentences each):
        [0.80, 0.80, 0.80, 0.80] -> 80% (70% - 90%)   sentences agree -> narrow
        [0.20, 0.90, 0.30, 0.90] -> 58% (24% - 91%)   mixed -> wide
    """
    p = np.asarray(sentence_probs, dtype=float)
    n = len(p)
    if n == 0:   # e.g. every sentence was under 5 words - nothing to score
        return {'score': np.nan, 'low': np.nan, 'high': np.nan, 'n_sentences': 0}
    mean = float(p.mean())
    half_width = z * np.sqrt((p.var() + min_variance) / n)
    return {'score': mean,
            'low': max(0.0, mean - half_width),
            'high': min(1.0, mean + half_width),
            'n_sentences': n}


def likelihood_band(probability):
    """Map a probability to the mockup's label. Example: 0.67 -> 'Likely AI'."""
    for upper, name in BANDS:
        if probability <= upper:
            return name
    return BANDS[-1][1]


def documents_from_sentences(sent_df, sentence_probs):
    """Group scored sentences back into their documents (via document_id)
    and aggregate each document - exactly what the website will do."""
    rows = []
    for doc_id, idx in sent_df.groupby('document_id', sort=False).indices.items():
        first = sent_df.iloc[idx[0]]
        rows.append({'document_id': doc_id, 'label': int(first['label']),
                     'group_key': first['group_key'],
                     **aggregate_document(sentence_probs[idx])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def main(args):
    # Fixed seeds -> same random weights / shuffling every run (repeatable results).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load sentence splits (made from the leakage-safe document splits)
    # ------------------------------------------------------------------
    # Documents were split by prompt BEFORE sentence splitting, so sentences
    # of one essay/prompt can never land in two partitions. We re-check.
    data = Path(args.sentence_dir)
    train = load_sentences(data / 'drcat_train_sentences.csv')
    splits = {'validation': load_sentences(data / 'drcat_validation_sentences.csv'),
              'test': load_sentences(data / 'drcat_test_sentences.csv'),
              'hc3_external': load_sentences(data / 'hc3_external_test_sentences.csv')}
    assert set(train.group_key).isdisjoint(splits['validation'].group_key)
    assert set(train.group_key).isdisjoint(splits['test'].group_key)

    # Optional speed-up: train on a random subset of TRAIN sentences only.
    # Evaluation sets are never subsampled.
    if args.max_train_sentences and len(train) > args.max_train_sentences:
        train = train.sample(args.max_train_sentences, random_state=args.seed).reset_index(drop=True)
    print(f'Training on {len(train):,} sentences; labels {train.label.value_counts().to_dict()}')

    # ------------------------------------------------------------------
    # 2-3) Tokenizer (training sentences only) + encode everything
    # ------------------------------------------------------------------
    tokenizer = train_tokenizer(train.text.astype(str).tolist(), args.vocab_size, out)
    pad_id = tokenizer.token_to_id(PAD)
    print(f'Tokenizer trained: {tokenizer.get_vocab_size():,} tokens')
    tr_encoded = encode_sentences(train.text, tokenizer, args.max_len)
    tr_labels = train.label.to_numpy()
    encoded = {name: encode_sentences(df.text, tokenizer, args.max_len)
               for name, df in splits.items()}

    # ------------------------------------------------------------------
    # 4) Build the model with random weights
    # ------------------------------------------------------------------
    # Saved alongside the weights so the backend can rebuild it identically.
    model_config = dict(vocab_size=tokenizer.get_vocab_size(), max_len=args.max_len,
                        d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
                        ff_dim=args.ff_dim, dropout=args.dropout, pad_id=pad_id)
    device = pick_device()
    model = TransformerClassifier(**model_config).to(device)
    print(f'Device: {device}; model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ------------------------------------------------------------------
    # 5) Training set-up
    # ------------------------------------------------------------------
    loader = make_loader(tr_encoded, tr_labels, pad_id, args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = warmup_then_linear_decay(optimizer, max(1, int(0.1 * total_steps)), total_steps)

    # Class-balanced loss: DRCAT is ~39% AI (and human essays tend to have more
    # sentences). Weighting makes mistakes on the rarer class cost more.
    # Example: 6,000 human + 4,000 AI sentences -> weights 0.83 (human), 1.25 (AI)
    counts = np.bincount(tr_labels, minlength=2)
    weights = torch.tensor(len(tr_labels) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------------------
    # 6) Training loop
    # ------------------------------------------------------------------
    # One "epoch" = one full pass over every training sentence.
    y_val = splits['validation'].label.to_numpy()
    best_auc, best_path = -1.0, out / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()   # dropout on
        running = 0.0
        for step, (input_ids, attention_mask, labels) in enumerate(loader, 1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = labels.to(device)
            # Forward pass: predict, then measure how wrong we were.
            loss = loss_fn(model(input_ids, attention_mask), labels)
            # Backward pass: how much did each weight contribute to the error?
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cap update size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 200 == 0:
                # Average loss over the last 200 batches - should trend downwards.
                print(f'  epoch {epoch} step {step}/{len(loader)} loss={running / 200:.4f}', flush=True)
                running = 0.0

        # Keep the epoch with the best SENTENCE-level validation ROC-AUC, since
        # sentence scoring is the core requirement. Test/HC3 are never used here.
        val_p = predict_sentences(model, encoded['validation'], pad_id, args.batch_size * 2, device)
        val_auc = roc_auc_score(y_val, val_p)
        print(f'Epoch {epoch}: validation sentence ROC-AUC={val_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model_state': model.state_dict(), 'config': model_config}, best_path)

    # ------------------------------------------------------------------
    # 7) Score every sentence with the best model, then aggregate to documents
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state'])
    sent_probs = {name: predict_sentences(model, enc, pad_id, args.batch_size * 2, device)
                  for name, enc in encoded.items()}
    docs = {name: documents_from_sentences(splits[name], sent_probs[name]) for name in splits}

    # ------------------------------------------------------------------
    # 8) Thresholds from validation, then report every split at both levels
    # ------------------------------------------------------------------
    # Separate thresholds because a single sentence and an averaged document
    # behave differently. NOTE: the website's bands assume 0.5 separates
    # "Possibly AI" from "Likely AI" - if the tuned document threshold is far
    # from 0.5, discuss recalibrating the band boundaries.
    sent_threshold = choose_threshold(y_val, sent_probs['validation'])
    doc_threshold = choose_threshold(docs['validation'].label.to_numpy(),
                                     docs['validation'].score.to_numpy())
    name = 'transformer_from_scratch'

    # How to read the rows:
    #   validation   -> optimistic (used for epoch + threshold choice)
    #   test         -> honest score on DRCAT prompts never seen in training
    #   hc3_external -> honest score on a DIFFERENT kind of text; a big drop
    #                   means the model learned "DRCAT essay style" more than
    #                   "AI style"
    #   Req 13's 70% target -> check 'accuracy' on the test rows.
    metrics = []
    for split_name in ['validation', 'test', 'hc3_external']:
        s_df, d_df = splits[split_name], docs[split_name]
        for level, y, p, thr in [
            ('sentence', s_df.label.to_numpy(), sent_probs[split_name], sent_threshold),
            ('document', d_df.label.to_numpy(), d_df.score.to_numpy(), doc_threshold),
        ]:
            result = score(y, p, thr)
            metrics.append({'model': name, 'level': level, 'split': split_name,
                            'threshold_tuned_on_validation': thr, **result})
            print(f'{split_name:12s} {level:8s}: accuracy={result["accuracy"]:.3f} '
                  f'ROC-AUC={result["roc_auc"]:.3f} F1={result["f1_ai"]:.3f}')

        if split_name == 'validation':
            continue
        # Per-sentence predictions -> check highlighting quality, e.g. which
        # kinds of sentences get "Very likely AI" wrongly.
        sentence_out = s_df[['sentence_id', 'document_id', 'sentence_index', 'text', 'label',
                             'n_words', 'generator']].copy()
        sentence_out['prob_ai'] = sent_probs[split_name]
        sentence_out['band'] = sentence_out.prob_ai.map(likelihood_band)
        sentence_out.to_csv(out / f'{split_name}_sentence_predictions.csv', index=False)
        # Per-document scores + confidence ranges, as the results screen shows them.
        doc_out = d_df.copy()
        doc_out['band'] = doc_out.score.map(likelihood_band)
        doc_out['pred_ai'] = (doc_out.score >= doc_threshold).astype(int)
        doc_out.to_csv(out / f'{split_name}_document_predictions.csv', index=False)

    pd.DataFrame(metrics).to_csv(out / 'transformer_comparison.csv', index=False)
    print(f'Thresholds (validation only): sentence={sent_threshold:.2f}, document={doc_threshold:.2f}')
    print(f'Results saved to: {out.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sentence-dir', default='data/sentences',
                    help='folder with the *_sentences.csv files from sentence_splitter.py')
    ap.add_argument('--output-dir', default='results/transformer')
    # --- tokenizer ---
    ap.add_argument('--vocab-size', type=int, default=8000,
                    help='size of the BPE vocabulary we train')
    ap.add_argument('--max-len', type=int, default=64, help='max tokens per sentence')
    # --- model size (bigger = more capacity, but slower and overfits more easily) ---
    ap.add_argument('--d-model', type=int, default=256, help='vector size per token')
    ap.add_argument('--heads', type=int, default=4, help='attention heads (must divide d-model)')
    ap.add_argument('--layers', type=int, default=4, help='stacked encoder layers')
    ap.add_argument('--ff-dim', type=int, default=1024, help='hidden size of each feed-forward block')
    ap.add_argument('--dropout', type=float, default=0.1)
    # --- training ---
    ap.add_argument('--max-train-sentences', type=int, default=0,
                    help='0 = use all training sentences')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)   # higher than fine-tuning: weights start random
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())