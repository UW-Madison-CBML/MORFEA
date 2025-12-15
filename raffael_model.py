"""
Complete High-Quality ConvLSTM Autoencoder
- Uses true ConvLSTM (not regular LSTM)
- Complete Encoder (2D CNN + ConvLSTM) with flattened latents
- Complete Decoder (ConvLSTM + ConvTranspose)
- Optional Empty/Non-empty Classifier
- Works with 128x128 input images
- Latent format: (B, T, N) where N is flattened spatial dimensions
"""
import torch
import torch.nn as nn
from raffael_conv_lstm import ConvLSTM
from huggingface_hub import PyTorchModelHubMixin


class Encoder(nn.Module):
    """
    Encoder: 2D CNN spatial compression + ConvLSTM temporal modeling + flatten to (B, T, N)
    Output: z_seq (B, T, latent_size) and z_last (B, latent_size)
    """

    def __init__(self, input_channels=1, hidden_dim=256, num_layers=2, latent_size=4096):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_size = latent_size

        # Spatial convolution: process each frame separately
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.spatial_cnn = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            # Layer 2: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            # Layer 3: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        # ConvLSTM: process temporal sequence
        # Input: (B, T, 256, 16, 16)
        # Output: (B, T, hidden_dim, 16, 16)
        self.convlstm = ConvLSTM(
            input_dim=256,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )

        # Dropout before latent compression
        self.dropout = nn.Dropout(0.1)

        # Linear layer to compress spatial latent to fixed size
        # Input: (B*T, hidden_dim * 16 * 16)
        # Output: (B*T, latent_size)
        self.latent_compress = nn.Linear(hidden_dim * 16 * 16, latent_size)

    def forward(self, x):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (128x128)

        Returns:
            z_seq: (B, T, latent_size) - compressed latent sequence
            z_last: (B, latent_size) - last timestep compressed latent
        """
        B, T, C, H, W = x.shape

        # Spatial compression: process each frame separately
        x = x.view(B * T, C, H, W)  # (B*T, 1, 128, 128)
        x = self.spatial_cnn(x)      # (B*T, 256, 16, 16)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)  # (B, T, 256, 16, 16)

        # ConvLSTM processes temporal sequence
        lstm_out, _ = self.convlstm(x)  # list of (B, T, hidden_dim, 16, 16)
        h_seq = lstm_out[0]             # (B, T, hidden_dim, 16, 16)

        # Flatten and compress spatial dimensions with linear layer
        B, T, C, H, W = h_seq.shape
        h_flat = h_seq.view(B * T, C * H * W)  # (B*T, hidden_dim * 16 * 16)
        h_flat = self.dropout(h_flat)  # Apply dropout
        z_compressed = self.latent_compress(h_flat)  # (B*T, latent_size)
        z_seq = z_compressed.view(B, T, self.latent_size)  # (B, T, latent_size)

        # Take last timestep
        z_last = z_seq[:, -1]  # (B, latent_size)

        return z_seq, z_last


class Decoder(nn.Module):
    """
    Decoder: Linear expansion + ConvLSTM temporal decoding + ConvTranspose spatial reconstruction
    Input: z_seq (B, T, latent_size)
    Output: x_rec (B, T, 1, 128, 128)
    """

    def __init__(self, seq_len, latent_size=4096, latent_dim=256, hidden_dim=128, num_layers=2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.latent_size = latent_size

        # Linear layer to expand compressed latent to spatial dimensions
        # Input: (B*T, latent_size)
        # Output: (B*T, latent_dim * 16 * 16)
        self.latent_expand = nn.Linear(latent_size, latent_dim * 16 * 16)

        # ConvLSTM decodes temporal dimension
        self.convlstm = ConvLSTM(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )

        # Spatial decoding: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.spatial_decoder = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final output layer
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assume pixels normalized to [0,1]
        )

    def forward(self, z_seq):
        """
        Args:
            z_seq: (B, T, latent_size) - compressed latent sequence from encoder

        Returns:
            x_rec: (B, T, 1, 128, 128) - reconstructed video sequence
        """
        B, T, L = z_seq.shape

        # Expand compressed latent to spatial dimensions
        z_flat = z_seq.view(B * T, L)  # (B*T, latent_size)
        z_expanded = self.latent_expand(z_flat)  # (B*T, latent_dim * 16 * 16)
        z_spatial = z_expanded.view(B, T, self.latent_dim, 16, 16)  # (B, T, latent_dim, 16, 16)

        # ConvLSTM decodes temporal dimension
        lstm_out, _ = self.convlstm(z_spatial)  # list of (B, T, hidden_dim, 16, 16)
        h_seq = lstm_out[0]                 # (B, T, hidden_dim, 16, 16)

        # Spatial decoding: process each timestep separately
        B, T, C, H, W = h_seq.shape
        h_seq = h_seq.view(B * T, C, H, W)  # (B*T, hidden_dim, 16, 16)
        x_rec = self.spatial_decoder(h_seq)  # (B*T, 1, 128, 128)
        x_rec = x_rec.view(B, T, 1, 128, 128)  # (B, T, 1, 128, 128)

        return x_rec


class LatentClassifier(nn.Module):
    """
    Empty / Non-empty Well Classifier
    Classifies based on last timestep latent
    """

    def __init__(self, latent_size=4096, num_classes=2, dropout=0.3):
        super(LatentClassifier, self).__init__()

        self.head = nn.Sequential(
            # Classification head - input is already flattened (B, latent_size)
            nn.Linear(latent_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, z_last):
        """
        Args:
            z_last: (B, latent_size) - last timestep compressed latent

        Returns:
            logits: (B, num_classes) - classification logits
        """
        return self.head(z_last)


class ConvLSTMAutoencoder(nn.Module, PyTorchModelHubMixin):
    """
    Complete ConvLSTM Autoencoder
    Includes Encoder, Decoder, and optional Classifier
    Compatible with HuggingFace Hub
    Works with 128x128 images
    """

    def __init__(
        self,
        seq_len=20,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=True,
        num_classes=2
    ):
        super(ConvLSTMAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.use_classifier = use_classifier
        self.encoder_hidden_dim = encoder_hidden_dim
        self.latent_size = latent_size

        # Core components
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            latent_size=latent_size
        )

        self.decoder = Decoder(
            seq_len=seq_len,
            latent_size=latent_size,
            latent_dim=encoder_hidden_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers
        )

        # Optional classifier
        if use_classifier:
            self.classifier = LatentClassifier(
                latent_size=latent_size,
                num_classes=num_classes
            )

    def forward(self, x, return_all=False):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (128x128)
            return_all: whether to return all intermediate results

        Returns:
            Tuple of (reconstruction, lat_vec_seq) where:
                - reconstruction: (B, T, 1, 128, 128) - reconstructed video
                - lat_vec_seq: (B, T, latent_size) - compressed latent sequence

            If return_all is True, returns dict with keys:
                - reconstruction: (B, T, 1, H, W) - reconstructed video
                - z_seq: (B, T, latent_size) - compressed latent sequence
                - z_last: (B, latent_size) - last timestep compressed latent
                - logits: (B, num_classes) - classification logits (if enabled)
        """
        # Encode
        z_seq, z_last = self.encoder(x)

        # Decode
        x_rec = self.decoder(z_seq)

        if return_all:
            # Build output dictionary
            output = {
                "reconstruction": x_rec,
                "z_seq": z_seq,
                "z_last": z_last,
            }

            # Optional classification
            if self.use_classifier:
                logits = self.classifier(z_last)
                output["logits"] = logits

            return output
        else:
            # Return tuple: (reconstruction, latent_vector)
            return x_rec, z_seq

    def encode(self, x):
        """Encode only, for extracting latent"""
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last

    def decode(self, z_seq):
        """Decode only, for reconstructing from latent"""
        return self.decoder(z_seq)
