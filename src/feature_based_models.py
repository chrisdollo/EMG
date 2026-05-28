import math

import torch
import torch.nn as nn


# ── Dataset constants ─────────────────────────────────────────────────────────
NUM_CLASSES   = 7
INPUT_SIZE    = 3192   # 50 libemg feature groups × 24 channels (= 1200 scalars * varies)
SEQ_LEN       = 26    # 26 windows per rep (window_size=250, shift=50, signal=1500)


# ── 1. FeatureLSTM ────────────────────────────────────────────────────────────
#
# Bidirectional LSTM over the 26-step feature sequence.
# Reads forward and backward, concatenates both directions at the last timestep,
# then classifies.

class FeatureLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
                 hidden_size=128, num_layers=2, dropout_rate=0.3,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_size, num_classes),
        )

    def forward(self, x):
        # x: (B, seq_len, input_size)
        out, _ = self.lstm(x)        # (B, seq_len, hidden * dirs)
        x = out[:, -1, :]            # last timestep
        return self.classifier(x)


# ── 2. FeatureGRU ─────────────────────────────────────────────────────────────
#
# Lighter alternative to LSTM — single gating mechanism instead of three.
# Often matches LSTM accuracy with fewer parameters and faster training.

class FeatureGRU(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
                 hidden_size=128, num_layers=2, dropout_rate=0.3,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        x = out[:, -1, :]
        return self.classifier(x)


# ── 3. FeatureTransformer ─────────────────────────────────────────────────────
#
# Self-attention encoder over the 26-step feature sequence.
# Sinusoidal positional encoding is added so the model is sequence-aware.
# Mean-pools the encoder output across time before classification.

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeatureTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                 dropout_rate=0.3, seq_len=SEQ_LEN):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=max(seq_len, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)       # (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)          # (B, seq_len, d_model)
        x = x.mean(dim=1)            # mean pool over time
        return self.classifier(x)
