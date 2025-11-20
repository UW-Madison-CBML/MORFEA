"""
Complete High-Quality ConvLSTM Autoencoder
- Uses true ConvLSTM (not regular LSTM)
- Complete Encoder (2D CNN + ConvLSTM)
- Complete Decoder (ConvLSTM + ConvTranspose)
- Optional Empty/Non-empty Classifier
- Maximum quality configuration, no computational savings
"""
import torch
import torch.nn as nn
from conv_lstm import ConvLSTM


class Encoder(nn.Module):
    """
    Encoder: 2D CNN spatial compression + ConvLSTM temporal modeling
    Output: z_seq (B, T, C, H, W) and z_last (B, C, H, W)
    """
    
    def __init__(self, input_channels=1, hidden_dim=256, num_layers=2):
        super(Encoder, self).__init__()
        
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
    
    def forward(self, x):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence
        
        Returns:
            z_seq: (B, T, hidden_dim, H_latent, W_latent) - full temporal sequence latent
            z_last: (B, hidden_dim, H_latent, W_latent) - last timestep latent
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
        
        # Version A: keep full temporal sequence latent
        z_seq = h_seq  # (B, T, hidden_dim, 16, 16)
        
        # Version B: take only last timestep
        z_last = h_seq[:, -1]  # (B, hidden_dim, 16, 16)
        
        return z_seq, z_last


class Decoder(nn.Module):
    """
    Decoder: ConvLSTM temporal decoding + ConvTranspose spatial reconstruction
    Input: z_seq (B, T, C, H, W)
    Output: x_rec (B, T, 1, 128, 128)
    """
    
    def __init__(self, seq_len, latent_dim=256, hidden_dim=128, num_layers=2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        
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
            z_seq: (B, T, latent_dim, H_latent, W_latent) - latent sequence from encoder
        
        Returns:
            x_rec: (B, T, 1, 128, 128) - reconstructed video sequence
        """
        # ConvLSTM decodes temporal dimension
        lstm_out, _ = self.convlstm(z_seq)  # list of (B, T, hidden_dim, 16, 16)
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
    
    def __init__(self, latent_dim=256, num_classes=2, dropout=0.3):
        super(LatentClassifier, self).__init__()
        
        self.head = nn.Sequential(
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),  # (C, H, W) -> (C, 1, 1)
            nn.Flatten(),              # (C,)
            
            # Classification head
            nn.Linear(latent_dim, 512),
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
            z_last: (B, latent_dim, H, W) - last timestep latent
        
        Returns:
            logits: (B, num_classes) - classification logits
        """
        return self.head(z_last)


class ConvLSTMAutoencoder(nn.Module):
    """
    Complete ConvLSTM Autoencoder
    Includes Encoder, Decoder, and optional Classifier
    """
    
    def __init__(
        self,
        seq_len=20,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=True,
        num_classes=2
    ):
        super(ConvLSTMAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.use_classifier = use_classifier
        
        # Core components
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers
        )
        
        self.decoder = Decoder(
            seq_len=seq_len,
            latent_dim=encoder_hidden_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers
        )
        
        # Optional classifier
        if use_classifier:
            self.classifier = LatentClassifier(
                latent_dim=encoder_hidden_dim,
                num_classes=num_classes
            )
    
    def forward(self, x, return_all=False):
        """
        Args:
            x: (B, T, 1, H, W) - input video sequence
            return_all: whether to return all intermediate results
        
        Returns:
            dict with keys:
                - reconstruction: (B, T, 1, H, W) - reconstructed video
                - z_seq: (B, T, C, H, W) - full latent sequence
                - z_last: (B, C, H, W) - last timestep latent
                - logits: (B, num_classes) - classification logits (if enabled)
        """
        # Encode
        z_seq, z_last = self.encoder(x)
        
        # Decode
        x_rec = self.decoder(z_seq)
        
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
    
    def encode(self, x):
        """Encode only, for extracting latent"""
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last
    
    def decode(self, z_seq):
        """Decode only, for reconstructing from latent"""
        return self.decoder(z_seq)

