import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dataset constants ─────────────────────────────────────────────────────────
NUM_CLASSES   = 7
NUM_CHANNELS  = 24
INPUT_SAMPLES = 1500


# ── 1. EEGNet ─────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS,
                 input_samples=INPUT_SAMPLES, dropout_rate=0.25,
                 kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_samples)
            x = self.separableConv(self.depthwiseConv(self.firstConv(dummy)))
            flat = x.shape[1] * x.shape[2] * x.shape[3]
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(flat, num_classes))

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return self.classify(x)


# ── 2. ShallowConvNet (Schirrmeister et al., 2017) ────────────────────────────
#
# Architecture:
#   temporal conv (1×25) → spatial depthwise conv (C×1)
#   → square nonlinearity → average pool → log nonlinearity → FC
#
# Rationale: specifically designed for EEG/EMG, leverages band-power features
# via the square+log pipeline (equivalent to log-power of filtered signal).

class ShallowConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS,
                 input_samples=INPUT_SAMPLES, dropout_rate=0.5, n_filters=40):
        super(ShallowConvNet, self).__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(n_filters)
        )
        # Depthwise spatial conv across all channels
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(num_channels, 1),
                      groups=n_filters, bias=False),
            nn.BatchNorm2d(n_filters)
        )
        self.pool    = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_samples)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            flat = x.flatten(1).shape[1]

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, num_classes))

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return self.classifier(x)


# ── 3. DeepConvNet (Schirrmeister et al., 2017) ───────────────────────────────
#
# Architecture:
#   Block 1: temporal conv (1×5) → spatial conv (C×1) → BN → ELU → MaxPool
#   Blocks 2–4: Conv(1×5) → BN → ELU → MaxPool, filters: 25→50→100→200
#   → FC
#
# Rationale: deeper hierarchy extracts progressively abstract temporal features.
# Standard deep-learning EEG/EMG baseline.

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS,
                 input_samples=INPUT_SAMPLES, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()

        def conv_block(in_f, out_f, kernel, pool, drop):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ELU(),
                nn.MaxPool2d(pool),
                nn.Dropout(p=drop)
            )

        # Block 1: separate temporal and spatial convolutions
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate)
        )
        self.block2 = conv_block(25,  50,  (1, 5), (1, 2), dropout_rate)
        self.block3 = conv_block(50,  100, (1, 5), (1, 2), dropout_rate)
        self.block4 = conv_block(100, 200, (1, 5), (1, 2), dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_samples)
            x = self.block4(self.block3(self.block2(self.block1(dummy))))
            flat = x.flatten(1).shape[1]

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ── 4. CNN_LSTM ────────────────────────────────────────────────────────────────
#
# Architecture:
#   Spatial CNN (C×1 depthwise) → temporal downsampling → 2-layer LSTM → FC
#
# Rationale: the CNN extracts spatial electrode features; the LSTM models
# sequential temporal dynamics across the gesture. Temporal pooling (×15)
# reduces the 1500-step sequence to 100 steps before the LSTM.

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS,
                 input_samples=INPUT_SAMPLES, dropout_rate=0.25,
                 cnn_filters=32, lstm_hidden=64, lstm_layers=2):
        super(CNN_LSTM, self).__init__()

        # Spatial mixing across electrode channels
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, cnn_filters, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(cnn_filters),
            nn.ELU(),
            nn.Dropout(p=dropout_rate)
        )
        # Output: (B, cnn_filters, 1, input_samples) → squeeze → (B, cnn_filters, input_samples)

        # Downsample time axis before LSTM  (1500 → 100)
        self.temporal_pool = nn.AvgPool1d(kernel_size=15)

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        x = self.spatial_conv(x)       # (B, 32, 1, 1500)
        x = x.squeeze(2)               # (B, 32, 1500)
        x = self.temporal_pool(x)      # (B, 32, 100)
        x = x.permute(0, 2, 1)        # (B, 100, 32)
        x, _ = self.lstm(x)            # (B, 100, 64)
        x = x[:, -1, :]               # (B, 64)  — last time step
        return self.classifier(x)


# ── 5. EMG_TCN (Temporal Convolutional Network) ───────────────────────────────
#
# Architecture:
#   Spatial mixing (C×1 conv) → 4 dilated TCN residual blocks → global avg pool → FC
#   Dilations: 1, 2, 4, 8  |  Filters: 32→64→64→128
#
# Rationale: dilated convolutions expand the receptive field exponentially
# without increasing parameters, capturing both fine and coarse temporal
# structure. Well-suited for long EMG sequences. No vanishing gradient issues.

class _TCNBlock(nn.Module):
    """Single residual dilated causal conv block."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        # Causal padding: pad only the left side
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )
        self.pad    = pad
        self.resize = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, :x.size(2)]    # trim causal padding
        res = x if self.resize is None else self.resize(x)
        return F.elu(out + res)


class EMG_TCN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS,
                 input_samples=INPUT_SAMPLES, dropout_rate=0.25):
        super(EMG_TCN, self).__init__()

        # Spatial mixing across electrode channels
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        # Output: (B, 32, 1, 1500) → squeeze → (B, 32, 1500)

        # Stacked dilated TCN blocks
        specs = [(32, 64, 1), (64, 64, 2), (64, 128, 4), (128, 128, 8)]
        self.tcn = nn.Sequential(*[
            _TCNBlock(in_ch, out_ch, kernel_size=3, dilation=dil, dropout=dropout_rate)
            for in_ch, out_ch, dil in specs
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.spatial(x)    # (B, 32, 1, 1500)
        x = x.squeeze(2)       # (B, 32, 1500)
        x = self.tcn(x)        # (B, 128, 1500)
        return self.classifier(x)
