"""
ConvLSTM Autoencoder Model
- Frame-level CNN encoder/decoder
- LSTM for temporal modeling
- Optional classifier head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    """CNN encoder for individual frames"""
    def __init__(self, input_channels=1, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B, 128, 1, 1]
        )
        self.proj = nn.Linear(128, out_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - single frame
        Returns:
            (B, out_dim) - frame embedding
        """
        h = self.net(x).squeeze(-1).squeeze(-1)  # [B, 128]
        return self.proj(h)  # [B, out_dim]


class FrameDecoder(nn.Module):
    """CNN decoder for individual frames"""
    def __init__(self, in_dim=128, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, 8 * 8 * 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, 4, 2, 1),  # 64 -> 128
            nn.Sigmoid()
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, in_dim) - latent vector
        Returns:
            (B, C, H, W) - reconstructed frame
        """
        x = self.fc(z)  # [B, 8*8*128]
        x = x.view(-1, 128, 8, 8)  # [B, 128, 8, 8]
        return self.deconv(x)  # [B, C, 128, 128]


class ConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder
    - Encodes each frame with CNN
    - Models temporal dynamics with LSTM
    - Decodes back to frames
    - Optional classifier head
    """
    def __init__(
        self,
        seq_len=20,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=False,
        num_classes=3  # A, B, C 三个类别
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.use_classifier = use_classifier
        
        # Frame encoder: CNN to encode each frame
        self.frame_encoder = FrameEncoder(
            input_channels=input_channels,
            out_dim=encoder_hidden_dim
        )
        
        # Encoder LSTM: processes frame embeddings
        self.encoder_lstm = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=encoder_layers,
            batch_first=True
        )
        
        # Decoder LSTM: generates latent sequence
        self.decoder_lstm = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=decoder_layers,
            batch_first=True
        )
        
        # Frame decoder: CNN to decode each frame
        self.frame_decoder = FrameDecoder(
            in_dim=decoder_hidden_dim,
            output_channels=input_channels
        )
        
        # Optional classifier head
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(decoder_hidden_dim // 2, num_classes)
            )
    
    def encode(self, x):
        """
        Encode input sequence to latent space
        Args:
            x: (B, T, C, H, W) - input video sequence
        Returns:
            z_seq: (B, T, hidden_dim) - latent sequence
            z_last: (B, hidden_dim) - last latent vector
        """
        B, T, C, H, W = x.shape
        
        # Encode each frame
        x_flat = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        frame_embeds = self.frame_encoder(x_flat)  # [B*T, encoder_hidden_dim]
        frame_embeds = frame_embeds.view(B, T, -1)  # [B, T, encoder_hidden_dim]
        
        # Process with encoder LSTM
        z_seq, _ = self.encoder_lstm(frame_embeds)  # [B, T, encoder_hidden_dim]
        
        # Get last latent
        z_last = z_seq[:, -1, :]  # [B, encoder_hidden_dim]
        
        return z_seq, z_last
    
    def decode(self, z_seq):
        """
        Decode latent sequence to reconstructed frames
        Args:
            z_seq: (B, T, hidden_dim) - latent sequence
        Returns:
            (B, T, C, H, W) - reconstructed video sequence
        """
        B, T, _ = z_seq.shape
        
        # Process with decoder LSTM
        h_dec, _ = self.decoder_lstm(z_seq)  # [B, T, decoder_hidden_dim]
        
        # Decode each frame
        recon_list = []
        for t in range(T):
            frame_latent = h_dec[:, t, :]  # [B, decoder_hidden_dim]
            frame_recon = self.frame_decoder(frame_latent)  # [B, C, H, W]
            recon_list.append(frame_recon)
        
        recon = torch.stack(recon_list, dim=1)  # [B, T, C, H, W]
        return recon
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, T, C, H, W) - input video sequence
        Returns:
            dict with:
                - reconstruction: (B, T, C, H, W)
                - z_seq: (B, T, hidden_dim)
                - z_last: (B, hidden_dim)
                - logits: (B, num_classes) [if use_classifier]
        """
        # Encode
        z_seq, z_last = self.encode(x)
        
        # Decode
        reconstruction = self.decode(z_seq)
        
        # Build output dict
        output = {
            "reconstruction": reconstruction,
            "z_seq": z_seq,
            "z_last": z_last
        }
        
        # Add classifier output if enabled
        if self.use_classifier:
            logits = self.classifier(z_last)  # [B, num_classes]
            output["logits"] = logits
        
        return output

