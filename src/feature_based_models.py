import torch
import torch.nn as nn


NUM_CLASSES = 7
INPUT_SIZE  = 3192   # 50 libemg features × 24 channels per window


# ── FeatureMLP ────────────────────────────────────────────────────────────────
#
# Classifies a single window's feature vector with a 3-layer MLP. Given 2D input
# (B, feat) it classifies per window — the runner trains on all windows and
# majority-votes each rep's windows to a label (putEMG paper protocol). If given
# 3D input (B, n_wins, feat) it falls back to mean-pooling the windows first.

class FeatureMLP(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
                 hidden_size=512, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),           # normalise across 3192 features
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)   # (B, 26, 3192) → (B, 3192)
        return self.net(x)
