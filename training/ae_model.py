import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

    def __init__(self, input_channels=1, num_layers=2, latent_size=4096, use_lstm=True):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        self.spatial_cnn = nn.Sequential(
            # 128 -> 64 
            ResidualBlock(input_channels, 32, downsample=True),

            # 64 -> 32 
            ResidualBlock(32, 64, downsample=True),

            # 32 -> 16 
            ResidualBlock(64, 64, downsample=True),
        )

        self.use_lstm = use_lstm
        """if not self.use_convlstm:
            self.convlstm = None 
        else:
            self.convlstm = ConvLSTM(
                input_dim=256,
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
                num_layers=num_layers,
                batch_first=True,
                return_all_layers=False
                )"""
        
        self.dropout = nn.Dropout(0.1)

        self.latent_compress = nn.Linear(64 * 16 * 16, latent_size)
        if self.use_lstm:
            self.lstm_enc = nn.LSTM(latent_size, latent_size, batch_first=True)
        else:
            self.lstm_enc = None

        self.lin1 = nn.Linear(latent_size,latent_size)



    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)  # (B*T, 1, H, W)
        if H != 128 or W != 128:
            x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)

        x = self.spatial_cnn(x)      # (B*T, 256, 16, 16)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)  # (B, T, 256, 16, 16)

        """# ConvLSTM processes temporal sequence
        if(self.use_convlstm):
            lstm_out, _ = self.convlstm(x)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]             # (B, T, hidden_dim, 16, 16)
        else:
            h_seq = x # just pass it forward if not"""
        h_seq = x

        # Flatten and compress spatial dimensions with linear layer
        B, T, C, H, W = h_seq.shape
        h_flat = h_seq.view(B, T, C * H * W)  # Linear just works on bottom most dim
        z_compressed = F.relu(self.latent_compress(h_flat))
        if self.use_lstm:
            z_compressed, _ = self.lstm_enc(z_compressed)
        z_compressed = self.dropout(z_compressed)
        z_compressed = self.lin1(z_compressed)
        z_seq = z_compressed.view(B, T, self.latent_size)  


        return z_seq


class Decoder(nn.Module):

    def __init__(self, latent_size=4096, num_layers=2, use_lstm=True):
        super(Decoder, self).__init__()
        self.latent_size = latent_size

        self.latent_expand = nn.Linear(latent_size, 64 * 16 * 16)

        self.use_lstm = use_lstm
        """if not self.use_convlstm:
            self.convlstm = None 
        else:
            self.convlstm = ConvLSTM(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
                num_layers=num_layers,
                batch_first=True,
                return_all_layers=False
            )"""
        if self.use_lstm:
            self.lstm_dec = nn.LSTM(latent_size, latent_size, batch_first=True)
        else:
            self.lstm_dec = None
    
        self.dropout = nn.Dropout(0.1)
        self.spatial_decoder = nn.Sequential(
            # 16 -> 32 
            ResidualUpBlock(64, 64),

            # 32 -> 64 
            ResidualUpBlock(64, 64),

            # 64 -> 128 
            ResidualUpBlock(64, 32),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assume pixels normalized to [0,1]
        )
        self.lin1 = nn.Linear(latent_size,latent_size)

    def forward(self, z_seq):
        B, T, L = z_seq.shape

        z_flat = z_seq # linear works on bottom most dim
        z_flat = F.relu(self.lin1(z_flat))
        if self.use_lstm:
            z_flat, _ = self.lstm_dec(z_flat)
        z_flat = self.dropout(z_flat) 
        z_expanded = F.relu(self.latent_expand(z_flat)) 
        z_spatial = z_expanded.view(B, T, 64, 16, 16)  

        """# ConvLSTM decodes temporal dimension
        if(self.use_convlstm):
            lstm_out, _ = self.convlstm(z_spatial)  # list of (B, T, hidden_dim, 16, 16)
            h_seq = lstm_out[0]                 # (B, T, hidden_dim, 16, 16)
        else:
            h_seq = z_spatial # just pass it forward"""
        h_seq = z_spatial
        # Spatial decoding: process each timestep separately
        B, T, C, H, W = h_seq.shape
        h_seq = h_seq.view(B * T, C, H, W)  # (B*T, hidden_dim, 16, 16)
        x_rec = self.spatial_decoder(h_seq)  # (B*T, 1, 128, 128)
        x_rec = x_rec.view(B, T, 1, 128, 128)  # (B, T, 1, 128, 128)

        return x_rec



class ConvLSTMAutoencoder(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        config=None,
        input_channels=1,
        encoder_layers=2,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=True,
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=0.1,
        use_lstm=True,
        use_residual=True,
        use_batchnorm=True
        ):
        super(ConvLSTMAutoencoder, self).__init__()
        self.use_classifier = use_classifier
        self.latent_size = latent_size
        self.use_latent_split = use_latent_split
        # Store ablation settings for reproducibility
        self.dropout_rate = dropout_rate
        self.use_lstm = use_lstm
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        if(config != None):
            if isinstance(config, dict):
                self.use_classifier = config.get('use_classifier', use_classifier)
                self.latent_size = config.get('latent_size', latent_size)
                self.use_latent_split = config.get('use_latent_split', use_latent_split)
                self.dropout_rate = config.get('dropout_rate', dropout_rate)
                self.use_lstm = config.get('use_lstm', use_lstm)
                self.use_residual = config.get('use_residual', use_residual)
                self.use_batchnorm = config.get('use_batchnorm', use_batchnorm)
            else:
                self.use_classifier = config.use_classifier
                self.latent_size = config.latent_size
                self.use_latent_split = config.use_latent_split
                self.dropout_rate = config.dropout_rate
                self.use_lstm = config.use_lstm
                self.use_residual = config.use_residual
                self.use_batchnorm = config.use_batchnorm

        self.encoder = Encoder(
            latent_size=self.latent_size,
            use_lstm=self.use_lstm
        )

        self.decoder = Decoder(
            latent_size=self.latent_size,
            use_lstm=self.use_lstm
        )


    def forward(self, x, return_all=False, hidden=None):
        B, T, C, orig_H, orig_W = x.shape


        z_seq = self.encoder(x)

        x_rec = self.decoder(z_seq)

        # Resize back to original input size if needed
        if orig_H != 128 or orig_W != 128:
            x_rec_flat = x_rec.view(B * T, C, 128, 128)
            x_rec_flat = torch.nn.functional.interpolate(x_rec_flat, size=(orig_H, orig_W), mode='bilinear', align_corners=True)
            x_rec = x_rec_flat.view(B, T, C, orig_H, orig_W)

        return x_rec, z_seq

    def encode(self, x):
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last

    def decode(self, z_seq):
        return self.decoder(z_seq)
