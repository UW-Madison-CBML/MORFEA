"""
Complete High-Quality ConvLSTM Autoencoder
- Uses true ConvLSTM (not regular LSTM)
- Complete Encoder (2D CNN + ConvLSTM) with flattened latents
- Complete Decoder (ConvLSTM + ConvTranspose)
- Optional Empty/Non-empty Classifier
- Works with 128x128 input images
- Latent format: (B, T, N) where N is flattened spatial dimensions
- Includes ResNet-style residual connections in CNN layers
"""
import torch
import torch.nn as nn
from raffael_conv_lstm import ConvLSTM
from huggingface_hub import PyTorchModelHubMixin


class ResidualBlock(nn.Module):
    """
    Residual block for encoder with optional downsampling
    Supports ablation: can disable residual connections and batch normalization
    """
    def __init__(self, in_channels, out_channels, downsample=False, use_residual=True, use_batchnorm=True):
        super(ResidualBlock, self).__init__()

        self.use_residual = use_residual
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Projection shortcut if channels change or downsampling (only if using residual)
        if use_residual and (in_channels != out_channels or downsample):
            shortcut_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if use_batchnorm:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x) if self.use_residual else 0

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            out += identity
        out = self.relu(out)

        return out


class ResidualUpBlock(nn.Module):
    """
    Residual block for decoder with upsampling
    Supports ablation: can disable residual connections and batch normalization
    """
    def __init__(self, in_channels, out_channels, use_residual=True, use_batchnorm=True):
        super(ResidualUpBlock, self).__init__()

        self.use_residual = use_residual

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Shortcut with upsampling (only if using residual)
        if use_residual:
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x) if self.use_residual else 0

        out = self.upsample(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn2(out)

        if self.use_residual:
            out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """
    Encoder: 2D CNN spatial compression + optional ConvLSTM temporal modeling + flatten to (B, T, N)
    Output: z_seq (B, T, latent_size) and z_last (B, latent_size)
    Supports ablation: dropout rate, ConvLSTM on/off, residual connections, batch normalization
    """

    def __init__(self, input_channels=1, hidden_dim=256, num_layers=2, latent_size=4096,
                 dropout_rate=0.1, use_convlstm=True, use_residual=True, use_batchnorm=True):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.use_convlstm = use_convlstm

        # Spatial convolution with residual connections: process each frame separately
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.spatial_cnn = nn.Sequential(
            # Layer 1: 128 -> 64 (with downsampling)
            ResidualBlock(input_channels, 64, downsample=True, use_residual=use_residual, use_batchnorm=use_batchnorm),

            # Layer 2: 64 -> 32 (with downsampling)
            ResidualBlock(64, 128, downsample=True, use_residual=use_residual, use_batchnorm=use_batchnorm),

            # Layer 3: 32 -> 16 (with downsampling)
            ResidualBlock(128, 256, downsample=True, use_residual=use_residual, use_batchnorm=use_batchnorm),
        )

        if use_convlstm:
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
            # Compress from hidden_dim * 16 * 16
            compress_size = hidden_dim * 16 * 16
        else:
            # No ConvLSTM - just pass through spatial features
            self.convlstm = None
            # Compress from 256 * 16 * 16 (spatial CNN output)
            compress_size = 256 * 16 * 16

        # Dropout before latent compression
        self.dropout = nn.Dropout(dropout_rate)

        # Linear layer to compress spatial latent to fixed size
        # Input: (B*T, compress_size)
        # Output: (B*T, latent_size)
        self.latent_compress = nn.Linear(compress_size, latent_size)

    def forward(self, x):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (any size, will be resized to 128x128)

        Returns:
            z_seq: (B, T, latent_size) - compressed latent sequence
            z_last: (B, latent_size) - last timestep compressed latent
        """
        B, T, C, H, W = x.shape

        # Resize to 128x128 if needed
        x = x.view(B * T, C, H, W)  # (B*T, 1, H, W)
        if H != 128 or W != 128:
            x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)

        # Spatial compression: process each frame separately
        x = self.spatial_cnn(x)      # (B*T, 256, 16, 16)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)  # (B, T, 256, 16, 16)

        if self.use_convlstm:
            # ConvLSTM processes temporal sequence
            lstm_out, _ = self.convlstm(x)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]             # (B, T, hidden_dim, 16, 16)
        else:
            # No temporal processing - just pass through spatial features
            h_seq = x  # (B, T, 256, 16, 16)

        # Flatten and compress spatial dimensions with linear layer
        B, T, C, H, W = h_seq.shape
        h_flat = h_seq.view(B * T, C * H * W)  # (B*T, C * 16 * 16)
        h_flat = self.dropout(h_flat)  # Apply dropout
        z_compressed = self.latent_compress(h_flat)  # (B*T, latent_size)
        z_seq = z_compressed.view(B, T, self.latent_size)  # (B, T, latent_size)

        # Take last timestep
        z_last = z_seq[:, -1]  # (B, latent_size)

        return z_seq, z_last


class Decoder(nn.Module):
    """
    Decoder: Linear expansion + optional ConvLSTM temporal decoding + ConvTranspose spatial reconstruction
    Input: z_seq (B, T, latent_size)
    Output: x_rec (B, T, 1, 128, 128)
    Supports ablation: ConvLSTM on/off, residual connections, batch normalization
    """

    def __init__(self, seq_len, latent_size=4096, latent_dim=256, hidden_dim=128, num_layers=2,
                 use_latent_split=False, use_convlstm=True, use_residual=True, use_batchnorm=True):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.use_latent_split = use_latent_split
        self.use_convlstm = use_convlstm

        # If using latent split, we only use half the latent for reconstruction
        effective_latent_size = latent_size // 2 if use_latent_split else latent_size

        # Linear layer to expand compressed latent to spatial dimensions
        # Input: (B*T, effective_latent_size)
        # Output: (B*T, latent_dim * 16 * 16)
        self.latent_expand = nn.Linear(effective_latent_size, latent_dim * 16 * 16)

        if use_convlstm:
            # ConvLSTM decodes temporal dimension
            self.convlstm = ConvLSTM(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
                num_layers=num_layers,
                batch_first=True,
                return_all_layers=False
            )
            # Spatial decoder input channels
            spatial_input_channels = hidden_dim
        else:
            # No ConvLSTM - just pass through expanded latent
            self.convlstm = None
            # Spatial decoder input channels
            spatial_input_channels = latent_dim

        # Spatial decoding with residual connections: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.spatial_decoder = nn.Sequential(
            # 16 -> 32 (with upsampling)
            ResidualUpBlock(spatial_input_channels, 128, use_residual=use_residual, use_batchnorm=use_batchnorm),

            # 32 -> 64 (with upsampling)
            ResidualUpBlock(128, 64, use_residual=use_residual, use_batchnorm=use_batchnorm),

            # 64 -> 128 (with upsampling)
            ResidualUpBlock(64, 32, use_residual=use_residual, use_batchnorm=use_batchnorm),

            # Final output layer
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assume pixels normalized to [0,1]
        )

    def forward(self, z_seq, empty_well=False):
        """
        Args:
            z_seq: (B, T, latent_size) - compressed latent sequence from encoder
            empty_well: bool - whether this is an empty well (uses first half of latent)

        Returns:
            x_rec: (B, T, 1, 128, 128) - reconstructed video sequence
        """
        B, T, L = z_seq.shape

        # If using latent split, select which half to use
        if self.use_latent_split:
            if empty_well:
                z_seq = z_seq[:, :, :L//2]  # First half for empty wells
            else:
                z_seq = z_seq[:, :, L//2:]  # Second half for embryos

        # Expand compressed latent to spatial dimensions
        z_flat = z_seq.view(B * T, -1)  # (B*T, effective_latent_size)
        z_expanded = self.latent_expand(z_flat)  # (B*T, latent_dim * 16 * 16)
        z_spatial = z_expanded.view(B, T, self.latent_dim, 16, 16)  # (B, T, latent_dim, 16, 16)

        if self.use_convlstm:
            # ConvLSTM decodes temporal dimension
            lstm_out, _ = self.convlstm(z_spatial)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]                 # (B, T, hidden_dim, 16, 16)
        else:
            # No temporal processing - just pass through expanded latent
            h_seq = z_spatial  # (B, T, latent_dim, 16, 16)

        # Spatial decoding: process each timestep separately
        B, T, C, H, W = h_seq.shape
        h_seq = h_seq.view(B * T, C, H, W)  # (B*T, C, 16, 16)
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
    Supports ablation studies: dropout, ConvLSTM, residual connections, batch normalization
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
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=0.1,
        use_convlstm=True,
        use_residual=True,
        use_batchnorm=True
    ):
        super(ConvLSTMAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.use_classifier = use_classifier
        self.encoder_hidden_dim = encoder_hidden_dim
        self.latent_size = latent_size
        self.use_latent_split = use_latent_split
        # Store ablation settings for reproducibility
        self.dropout_rate = dropout_rate
        self.use_convlstm = use_convlstm
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm

        # Core components
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            latent_size=latent_size,
            dropout_rate=dropout_rate,
            use_convlstm=use_convlstm,
            use_residual=use_residual,
            use_batchnorm=use_batchnorm
        )

        self.decoder = Decoder(
            seq_len=seq_len,
            latent_size=latent_size,
            latent_dim=encoder_hidden_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            use_latent_split=use_latent_split,
            use_convlstm=use_convlstm,
            use_residual=use_residual,
            use_batchnorm=use_batchnorm
        )

        # Optional classifier
        if use_classifier:
            self.classifier = LatentClassifier(
                latent_size=latent_size,
                num_classes=num_classes
            )

    def forward(self, x, empty_well=False, return_all=False):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (any size, will be resized internally)
            empty_well: bool - whether this is an empty well (for latent split)
            return_all: whether to return all intermediate results

        Returns:
            Tuple of (reconstruction, lat_vec_seq) where:
                - reconstruction: (B, T, 1, H, W) - reconstructed video (same size as input)
                - lat_vec_seq: (B, T, latent_size or latent_size//2) - compressed latent sequence

            If return_all is True, returns dict with keys:
                - reconstruction: (B, T, 1, H, W) - reconstructed video
                - z_seq: (B, T, latent_size) - compressed latent sequence (full)
                - z_last: (B, latent_size) - last timestep compressed latent (full)
                - logits: (B, num_classes) - classification logits (if enabled)
        """
        B, T, C, orig_H, orig_W = x.shape

        # Encode (will resize to 128x128 internally)
        z_seq, z_last = self.encoder(x)

        # Decode (outputs 128x128)
        x_rec = self.decoder(z_seq, empty_well=empty_well)

        # Resize back to original input size if needed
        if orig_H != 128 or orig_W != 128:
            x_rec_flat = x_rec.view(B * T, C, 128, 128)
            x_rec_flat = torch.nn.functional.interpolate(x_rec_flat, size=(orig_H, orig_W), mode='bilinear', align_corners=True)
            x_rec = x_rec_flat.view(B, T, C, orig_H, orig_W)

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
            # If using latent split, return only the relevant half
            if self.use_latent_split:
                if empty_well:
                    return x_rec, z_seq[:, :, :self.latent_size//2]
                else:
                    return x_rec, z_seq[:, :, self.latent_size//2:]
            else:
                return x_rec, z_seq

    def encode(self, x):
        """Encode only, for extracting latent"""
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last

    def decode(self, z_seq, empty_well=False):
        """Decode only, for reconstructing from latent"""
        return self.decoder(z_seq, empty_well=empty_well)
