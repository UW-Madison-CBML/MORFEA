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
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut if channels change or downsampling
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResidualUpBlock(nn.Module):
    """
    Residual block for decoder with upsampling
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualUpBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.upsample(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """
    Encoder: 2D CNN spatial compression + ConvLSTM temporal modeling + flatten to (B, T, N)
    Output: z_seq (B, T, latent_size) and z_last (B, latent_size)
    """

    def __init__(self, input_channels=1, hidden_dim=256, num_layers=2, latent_size=4096, use_convlstm=True):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_size = latent_size

        # Spatial convolution with residual connections: process each frame separately
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.spatial_cnn = nn.Sequential(
            # Layer 1: 128 -> 64 (with downsampling)
            ResidualBlock(input_channels, 64, downsample=True),

            # Layer 2: 64 -> 32 (with downsampling)
            ResidualBlock(64, 128, downsample=True),

            # Layer 3: 32 -> 16 (with downsampling)
            ResidualBlock(128, 256, downsample=True),
        )

        if self.no_conv:
            self.convlstm = None 
        else:
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
        self.use_convlstm = use_convlstm

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

        # ConvLSTM processes temporal sequence
        if(self.use_convlstm):
            lstm_out, _ = self.convlstm(x)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]             # (B, T, hidden_dim, 16, 16)
        else:
            h_seq = x # just pass it forward if not

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

    def __init__(self, seq_len, latent_size=4096, latent_dim=256, hidden_dim=256, num_layers=2, use_convlstm=True):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.latent_size = latent_size

        # Linear layer to expand compressed latent to spatial dimensions
        # Input: (B*T, latent_size)
        # Output: (B*T, latent_dim * 16 * 16)
        self.latent_expand = nn.Linear(latent_size, latent_dim * 16 * 16)

        # ConvLSTM decodes temporal dimension
        if self.no_conv:
            self.convlstm = None 
        else:
            self.convlstm = ConvLSTM(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
                num_layers=num_layers,
                batch_first=True,
                return_all_layers=False
            )

        # Spatial decoding with residual connections: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.spatial_decoder = nn.Sequential(
            # 16 -> 32 (with upsampling)
            ResidualUpBlock(hidden_dim, 128),

            # 32 -> 64 (with upsampling)
            ResidualUpBlock(128, 64),

            # 64 -> 128 (with upsampling)
            ResidualUpBlock(64, 32),

            # Final output layer
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assume pixels normalized to [0,1]
        )
        self.use_convlstm = use_convlstm

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
        if(self.use_convlstm):
            lstm_out, _ = self.convlstm(z_spatial)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]                 # (B, T, hidden_dim, 16, 16)
        else:
            h_seq = z_spatial # just pass it forward

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
        config=None,
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
        if(config != None):
            # Handle config as dict (from HuggingFace) or object
            if isinstance(config, dict):
                self.seq_len = config.get('seq_len', seq_len)
                self.use_classifier = config.get('use_classifier', use_classifier)
                self.encoder_hidden_dim = config.get('encoder_hidden_dim', encoder_hidden_dim)
                self.latent_size = config.get('latent_size', latent_size)
                self.use_latent_split = config.get('use_latent_split', use_latent_split)
                self.dropout_rate = config.get('dropout_rate', dropout_rate)
                self.use_convlstm = config.get('use_convlstm', use_convlstm)
                self.use_residual = config.get('use_residual', use_residual)
                self.use_batchnorm = config.get('use_batchnorm', use_batchnorm)
            else:
                self.seq_len = config.seq_len
                self.use_classifier = config.use_classifier
                self.encoder_hidden_dim = config.encoder_hidden_dim
                self.latent_size = config.latent_size
                self.use_latent_split = config.use_latent_split
                self.dropout_rate = config.dropout_rate
                self.use_convlstm = config.use_convlstm
                self.use_residual = config.use_residual
                self.use_batchnorm = config.use_batchnorm

            # Core components
        self.encoder = Encoder(
            latent_size=self.latent_size,
            use_convlstm=self.use_convlstm
        )

        self.decoder = Decoder(
            seq_len=self.seq_len,
            latent_size=self.latent_size,
            use_convlstm=self.use_convlstm
        )

        # Optional classifier
        if use_classifier:
            self.classifier = LatentClassifier(
                latent_size=latent_size,
                num_classes=num_classes
            )

    def forward(self, x, return_all=False, hidden=None):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence (any size, will be resized internally)
            return_all: whether to return all intermediate results

        Returns:
            Tuple of (reconstruction, lat_vec_seq) where:
                - reconstruction: (B, T, 1, H, W) - reconstructed video (same size as input)
                - lat_vec_seq: (B, T, latent_size) - compressed latent sequence

            If return_all is True, returns dict with keys:
                - reconstruction: (B, T, 1, H, W) - reconstructed video
                - z_seq: (B, T, latent_size) - compressed latent sequence
                - z_last: (B, latent_size) - last timestep compressed latent
                - logits: (B, num_classes) - classification logits (if enabled)
        """
        B, T, C, orig_H, orig_W = x.shape

        # Encode (will resize to 128x128 internally)
        if return_all:
            z_seq, z_last, h_last_enc, c_last_enc = self.encoder(x, return_all=return_all)#, hidden_state=hidden_state['enc'] if hidden_state != None else None)

        else:
            z_seq, z_last = self.encoder(x)#, hidden_state=hidden_state['enc'] if hidden_state != None else None)


        # Decode (outputs 128x128)
        if return_all:
            x_rec, h_last_dec, c_last_dec = self.decoder(z_seq, return_all=return_all)#, hidden_state=hidden_state['dec'] if hidden_state != None else None)

        else:
            x_rec = self.decoder(z_seq)#, hidden_state=hidden_state['dec'] if hidden_state != None else None)

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
            return x_rec, z_seq

    def encode(self, x):
        """Encode only, for extracting latent"""
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last

    def decode(self, z_seq):
        """Decode only, for reconstructing from latent"""
        return self.decoder(z_seq)
