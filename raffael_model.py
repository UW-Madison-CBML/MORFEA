"""
Complete High-Quality ConvLSTM Autoencoder
- Uses true ConvLSTM (not regular LSTM)
- Complete Encoder (2D CNN + ConvLSTM)
- Complete Decoder (ConvLSTM + ConvTranspose)
- Optional Empty/Non-empty Classifier
- Maximum quality configuration, no computational savings
- Handles 500x500 input images
- Skip connections in encoder and decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from raffael_conv_lstm import ConvLSTM
from huggingface_hub import PyTorchModelHubMixin


class Encoder(nn.Module):
    """
    Encoder: 2D CNN spatial compression + ConvLSTM temporal modeling + Linear compression
    Output: z_seq (B, T, L) and z_last (B, L) where L is latent_size
    With skip connections between layers
    """

    def __init__(self, input_channels=1, hidden_dim=256, num_layers=2, latent_size=4000):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_size = latent_size

        # Spatial convolution layers with skip connections
        # 500x500 -> 250x250 -> 125x125 -> 62x62 -> 31x31 -> 15x15 -> 7x7

        # Layer 1: 500 -> 250
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2: 250 -> 125
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # Layer 3: 125 -> 62
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Layer 4: 62 -> 31
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)

        # Layer 5: 31 -> 15
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2)

        # Layer 6: 15 -> 7
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(2)

        # Skip connection projections
        self.skip_proj_32_64 = nn.Conv2d(32, 64, 1)
        self.skip_proj_64_128 = nn.Conv2d(64, 128, 1)

        # ConvLSTM: process temporal sequence
        # Input: (B, T, 128, 7, 7)
        # Output: (B, T, hidden_dim, 7, 7)
        self.convlstm = ConvLSTM(
            input_dim=128,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )

        # Dropout before latent compression
        self.dropout = nn.Dropout(0.1)

        # Linear layer to compress spatial latent to fixed size
        # Input: (B*T, hidden_dim * 7 * 7)
        # Output: (B*T, latent_size)
        self.latent_compress = nn.Linear(hidden_dim * 7 * 7, latent_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (500x500)

        Returns:
            z_seq: (B, T, latent_size) - compressed latent sequence
            z_last: (B, latent_size) - last timestep compressed latent
        """
        B, T, C, H, W = x.shape

        # Spatial compression with skip connections: process each frame separately
        x = x.view(B * T, C, H, W)  # (B*T, 1, 500, 500)

        # Layer 1
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))  # (B*T, 32, 250, 250)

        # Layer 2 with skip
        x_skip = F.avg_pool2d(x, 2)
        x_skip = self.skip_proj_32_64(x_skip)
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))  # (B*T, 64, 125, 125)
        x = x + x_skip

        # Layer 3 with skip
        x_skip = F.avg_pool2d(x, 2)
        x_skip = self.skip_proj_64_128(x_skip)
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))  # (B*T, 128, 62, 62)
        x = x + x_skip

        # Layer 4 with skip
        x_skip = F.avg_pool2d(x, 2)
        x = F.relu(self.pool4(self.bn4(self.conv4(x))))  # (B*T, 128, 31, 31)
        x = x + x_skip

        # Layer 5 with skip
        x_skip = F.avg_pool2d(x, 2)
        x = F.relu(self.pool5(self.bn5(self.conv5(x))))  # (B*T, 128, 15, 15)
        x = x + x_skip

        # Layer 6 with skip
        x_skip = F.avg_pool2d(x, 2)
        x = F.relu(self.pool6(self.bn6(self.conv6(x))))  # (B*T, 128, 7, 7)
        x = x + x_skip

        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)  # (B, T, 128, 7, 7)

        # ConvLSTM processes temporal sequence
        lstm_out, _ = self.convlstm(x)  # list of (B, T, hidden_dim, 7, 7)
        h_seq = lstm_out[0]             # (B, T, hidden_dim, 7, 7)

        # Flatten and compress spatial dimensions with linear layer
        B, T, C, H, W = h_seq.shape
        h_flat = h_seq.view(B * T, C * H * W)  # (B*T, hidden_dim * 7 * 7)
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
    Output: x_rec (B, T, 1, 500, 500)
    With skip connections between layers
    """

    def __init__(self, seq_len, latent_size=4000, latent_dim=256, hidden_dim=128, num_layers=2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.latent_size = latent_size

        # Linear layer to expand compressed latent to spatial dimensions
        # Input: (B*T, latent_size)
        # Output: (B*T, latent_dim * 7 * 7)
        self.latent_expand = nn.Linear(latent_size, latent_dim * 7 * 7)

        # ConvLSTM decodes temporal dimension
        self.convlstm = ConvLSTM(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )

        # Spatial decoding layers: 7x7 -> 15x15 -> 31x31 -> 62x62 -> 125x125 -> 250x250 -> 500x500
        # Layer 1
        self.conv1dec = nn.Conv2d(hidden_dim, 128, 3, stride=1)
        self.bn1dec = nn.BatchNorm2d(128)

        # Layer 2
        self.conv2dec = nn.Conv2d(128, 128, 3, stride=1)
        self.bn2dec = nn.BatchNorm2d(128)

        # Layer 3
        self.conv3dec = nn.Conv2d(128, 128, 3, stride=1)
        self.bn3dec = nn.BatchNorm2d(128)

        # Layer 4
        self.conv4dec = nn.Conv2d(128, 128, 3, stride=1)
        self.bn4dec = nn.BatchNorm2d(128)

        # Layer 5
        self.conv6dec = nn.Conv2d(128, 64, 3, stride=1)
        self.bn6dec = nn.BatchNorm2d(64)

        # Layer 6
        self.conv7dec = nn.Conv2d(64, 32, 3, stride=1)
        self.bn7dec = nn.BatchNorm2d(32)

        # Final output layer
        self.conv8dec = nn.Conv2d(32, 1, 3, padding=1)

        # Skip connection projections
        self.skip_proj_128_64 = nn.Conv2d(128, 64, 1)
        self.skip_proj_64_32 = nn.Conv2d(64, 32, 1)
    
    def forward(self, z_seq):
        """
        Args:
            z_seq: (B, T, latent_size) - compressed latent sequence from encoder

        Returns:
            x_rec: (B, T, 1, 500, 500) - reconstructed video sequence
        """
        B, T, L = z_seq.shape

        # Expand compressed latent to spatial dimensions
        z_flat = z_seq.view(B * T, L)  # (B*T, latent_size)
        z_expanded = self.latent_expand(z_flat)  # (B*T, latent_dim * 7 * 7)
        z_spatial = z_expanded.view(B, T, self.latent_dim, 7, 7)  # (B, T, latent_dim, 7, 7)

        # ConvLSTM decodes temporal dimension
        lstm_out, _ = self.convlstm(z_spatial)  # list of (B, T, hidden_dim, 7, 7)
        h_seq = lstm_out[0]                 # (B, T, hidden_dim, 7, 7)

        # Spatial decoding with skip connections: process each timestep separately
        B, T, C, H, W = h_seq.shape
        x = h_seq.view(B * T, C, H, W)  # (B*T, hidden_dim, 7, 7)

        # Layer 1 with skip
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn1dec(self.conv1dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = x + x_skip

        # Layer 2 with skip
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn2dec(self.conv2dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = x + x_skip

        # Layer 3 with skip
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn3dec(self.conv3dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = x + x_skip

        # Layer 4 with skip
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn4dec(self.conv4dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x = x + x_skip

        # Layer 5 with skip projection
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn6dec(self.conv6dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x_skip = self.skip_proj_128_64(x_skip)
        x = x + x_skip

        # Layer 6 with skip projection
        x_skip = x.clone()
        x = F.relu(F.interpolate(self.bn7dec(self.conv7dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
        x_skip = F.interpolate(x_skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        x_skip = self.skip_proj_64_32(x_skip)
        x = x + x_skip

        # Final output layer
        x = F.sigmoid(F.interpolate(self.conv8dec(x), size=(500, 500), mode='bilinear', align_corners=True))
        x_rec = x.view(B, T, 1, 500, 500)  # (B, T, 1, 500, 500)

        return x_rec


class LatentClassifier(nn.Module):
    """
    Empty / Non-empty Well Classifier
    Classifies based on last timestep latent
    """

    def __init__(self, latent_size=4000, num_classes=2, dropout=0.3):
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
    """

    def __init__(
        self,
        seq_len=20,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4000,
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
            x: (B, T, 1, H, W) - input video sequence (500x500)
            return_all: whether to return all intermediate results

        Returns:
            Tuple of (reconstruction, lat_vec_seq) where:
                - reconstruction: (B, T, 1, 500, 500) - reconstructed video
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
            # Return tuple like model.py: (reconstruction, latent_vector)
            return x_rec, z_seq
    
    def encode(self, x):
        """Encode only, for extracting latent"""
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last
    
    def decode(self, z_seq):
        """Decode only, for reconstructing from latent"""
        return self.decoder(z_seq)

