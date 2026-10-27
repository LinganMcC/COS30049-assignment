"""The sentence classifier, copied from transformer/train_transformer.py.

The class must match training exactly, because the saved weights are loaded
into it by parameter name. See train_transformer.py for the full explanation
of each layer; only inference is needed here.
"""
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """Token + position embeddings -> transformer encoder -> mean pool -> 2 logits."""

    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, ff_dim, dropout, pad_id):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # PyTorch's padding mask is True where a position should be IGNORED.
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)
