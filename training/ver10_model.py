"""
ConvLSTM Autoencoder — Ver10
=============================
Changes from Ver09:
    - Removed CPC prediction heads (cpc_predictors, predict_future)
    - Architecture is clean encoder-decoder only
    - Temporal loss moved entirely to losses.py (rolled InfoNCE)

Architecture:
    FrameEncoder (CNN) -> Encoder LSTM -> z_seq -> Decoder LSTM -> FrameDecoder
"""

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """
    CNN encoder for individual frames.
    Input:  (B, C, H, W)  — expects H=W=128
    Output: (B, out_dim)
    """

    def __init__(self, input_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),             # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                                 # -> (B, 128, 4, 4)
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(x).flatten(1))


class FrameDecoder(nn.Module):
    """
    CNN decoder for individual frames.
    Input:  (B, in_dim)
    Output: (B, C, 128, 128)
    """

    def __init__(self, in_dim: int = 256, output_channels: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # 8  -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).reshape(-1, 256, 8, 8))


class ConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder — Ver10

    Clean encoder-decoder. Temporal structure enforced by rolled InfoNCE
    in losses.py, not by any architectural component.

    Forward returns:
        reconstruction  (B, T, C, H, W)
        z_seq           (B, T, D)
        z_last          (B, D)
    """

    def __init__(
        self,
        seq_len: int = 50,
        input_channels: int = 1,
        encoder_hidden_dim: int = 256,
        encoder_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 2,
        dropout_rate: float = 0.0,
        use_classifier: bool = False,
        num_classes: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_classifier = use_classifier

        self.frame_encoder = FrameEncoder(input_channels, encoder_hidden_dim)

        self.encoder_lstm = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout_rate if encoder_layers > 1 else 0.0,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout_rate if decoder_layers > 1 else 0.0,
        )

        self.frame_decoder = FrameDecoder(decoder_hidden_dim, input_channels)

        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(encoder_hidden_dim // 2, num_classes),
            )

    def encode(self, x: torch.Tensor):
        """
        Args:   x (B, T, C, H, W)
        Returns z_seq (B, T, D), z_last (B, D)
        """
        B, T, C, H, W = x.shape
        embeds = self.frame_encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        z_seq, (h_n, _) = self.encoder_lstm(embeds)
        return z_seq, h_n[-1]

    def decode(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:   z_seq (B, T, D)
        Returns (B, T, C, H, W)
        """
        B, T, _ = z_seq.shape
        h, _ = self.decoder_lstm(z_seq)
        frames = self.frame_decoder(h.reshape(B * T, -1))
        return frames.reshape(B, T, *frames.shape[1:])

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:   x (B, T, C, H, W)
        Returns dict: reconstruction, z_seq, z_last, [logits]
        """
        z_seq, z_last = self.encode(x)
        out = {
            "reconstruction": self.decode(z_seq),
            "z_seq":  z_seq,
            "z_last": z_last,
        }
        if self.use_classifier:
            out["logits"] = self.classifier(z_last)
        return out
