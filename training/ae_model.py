import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F


class CellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CellLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.cell_forward = nn.LSTMCell(input_size, hidden_size)
        self.cell_backward = nn.LSTMCell(input_size, hidden_size)
        
    def forward(self, x, hx=None):
        x_seq = x.transpose(0, 1)
        seq_len, batch_size, _ = x_seq.size()
        
        if hx is None:
            h_f = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c_f = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            h_b = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c_b = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_f, c_f, h_b, c_b = hx

        forward_h, forward_c = [None] * seq_len, [None] * seq_len
        backward_h, backward_c = [None] * seq_len, [None] * seq_len

        for t in range(seq_len):
            h_f, c_f = self.cell_forward(x_seq[t], (h_f, c_f))
            forward_h[t] = h_f
            forward_c[t] = c_f
            
        for t in reversed(range(seq_len)):
            h_b, c_b = self.cell_backward(x_seq[t], (h_b, c_b))
            backward_h[t] = h_b
            backward_c[t] = c_b

        f_h_seq = torch.stack(forward_h, dim=0)
        f_c_seq = torch.stack(forward_c, dim=0)
        b_h_seq = torch.stack(backward_h, dim=0)
        b_c_seq = torch.stack(backward_c, dim=0)

        f_h_seq = f_h_seq.transpose(0, 1)
        f_c_seq = f_c_seq.transpose(0, 1)
        b_h_seq = b_h_seq.transpose(0, 1)
        b_c_seq = b_c_seq.transpose(0, 1)

        out_seq = torch.cat([f_h_seq, b_h_seq,f_c_seq, b_c_seq], dim=-1)

        return out_seq # B, T, 4 * hidden_size

         

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 =nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUpBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.upsample(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, input_channels=1, num_layers=2, latent_size=4096, hidden_channels=64, use_lstm=True, use_residual=True):
        super(Encoder, self).__init__()
        self.use_residual = use_residual
        self.latent_size = latent_size
        self.hidden_channels = hidden_channels
        #self.spatial_cnn = nn.Sequential(
            # 256 -> 128
            #ResidualBlock(input_channels, self.hidden_channels, downsample=True),

            # 128 -> 64 
            #ResidualBlock(self.hidden_channels, self.hidden_channels, downsample=True),
        self.resblock1 = ResidualBlock(input_channels, self.hidden_channels)

            # 64 -> 32 
        self.resblock2 = ResidualBlock(self.hidden_channels, self.hidden_channels)

            # 32 -> 16 
        self.resblock3 = ResidualBlock(self.hidden_channels, self.hidden_channels)

            # 16 -> 8
            #ResidualBlock(self.hidden_channels, self.hidden_channels, downsample=True),

        #)
        self.final_resolution = 16 #2 ** (8 - len([module for module in self.spatial_cnn.modules() if not isinstance(self.spatial_cnn, nn.Sequential)]))
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

        self.latent_compress = nn.Linear(self.hidden_channels * self.final_resolution * self.final_resolution, latent_size)
        if self.use_lstm:
            self.lstm_enc = nn.LSTM(latent_size, latent_size, batch_first=True, bidirectional=True)
        else:
            self.lstm_enc = None

        self.lin1 = nn.Linear(latent_size*2,latent_size*2)
        self.lin2 = nn.Linear(latent_size*2,latent_size)



    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)  # (B*T, 1, H, W)

        # x = self.spatial_cnn(x)      
        x = self.resblock3(self.resblock2(self.resblock1(x)))
        _, C2, H2, W2 = x.shape
        residual = x 
        x = x.view(B, T, C2, H2, W2)  

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
        z_compressed = F.relu(self.lin1(z_compressed)) # B, T, L*2
        z_compressed = self.lin2(z_compressed) # B, T, L, no relu so that latent space can be negative
        z_seq = z_compressed.view(B, T, self.latent_size)  


        return z_seq, residual

class Decoder(nn.Module):

    def __init__(self, latent_size=4096, num_layers=2, hidden_channels=64, initial_resolution=16,final_size=128, use_lstm=True, use_residual=True):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_channels = hidden_channels

        #self.latent_expand = nn.Linear(latent_size, self.hidden_channels * 16 * 16)

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
            self.lstm_dec = nn.LSTM(latent_size, latent_size, batch_first=True, bidirectional=True)
        else:
            self.lstm_dec = None
    
        self.dropout = nn.Dropout(0.1)
        #self.spatial_decoder = nn.Sequential(
            # 8 -> 16
            #ResidualUpBlock(self.hidden_channels, self.hidden_channels),

            # 16 -> 32 
        self.resblock1 = ResidualUpBlock(self.hidden_channels, self.hidden_channels)

            # 32 -> 64 
        self.resblock2 = ResidualUpBlock(self.hidden_channels, self.hidden_channels)

            # 64 -> 128 
        self.resblock3 = ResidualUpBlock(self.hidden_channels, self.hidden_channels)

            # 128 -> 256
            #ResidualUpBlock(self.hidden_channels, self.hidden_channels),


        self.out_conv = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1)
        #    nn.Sigmoid()
        #)
        
        self.initial_resolution = initial_resolution
        
        self.lin1 = nn.Linear(latent_size,2*latent_size)
        self.lin2 = nn.Linear(2*latent_size, latent_size)

        self.latent_expand = nn.Linear(latent_size*2, self.hidden_channels * self.initial_resolution * self.initial_resolution)
        self.final_size = final_size
        self.use_residual = use_residual
        self.bn = nn.BatchNorm(self.hidden_channels)
    def forward(self, z_seq, residual):
        B, T, L = z_seq.shape
        z_flat = F.relu(self.lin1(z_seq))
        z_flat = F.relu(self.lin2(z_flat))
        if self.use_lstm:
            z_flat, _ = self.lstm_dec(z_flat)
        z_flat = self.dropout(z_flat) 
        z_expanded = F.relu(self.latent_expand(z_flat)) 
        assert z_expanded.shape == (B, T, self.hidden_channels * (self.initial_resolution ** 2)), f"BAD z_expanded shape: {z_expanded.shape}"
        z_spatial = z_expanded.view(B, T, self.hidden_channels, self.initial_resolution, self.initial_resolution)  

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
        h_seq = self.bn(h_seq) # normalize h_seq so that the model can't try to make it as small as residual, so that residual signal slowly fades
        x_rec = self.out_conv(self.resblock3(self.resblock2(self.resblock1(h_seq + residual))))
        x_rec = F.sigmoid(x_rec)
        x_rec = x_rec.view(B, T, 1, self.final_size, self.final_size)  # (B, T, 1, 128, 128)

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
        use_batchnorm=True,
        hidden_channels=64
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
        self.hidden_channels = hidden_channels
        if(config != None):
            if isinstance(config, dict):
                self.use_classifier = config.get('use_classifier', use_classifier)
                self.latent_size = config.get('latent_size', latent_size)
                self.use_latent_split = config.get('use_latent_split', use_latent_split)
                self.dropout_rate = config.get('dropout_rate', dropout_rate)
                self.use_lstm = config.get('use_lstm', use_lstm)
                self.use_residual = config.get('use_residual', use_residual)
                self.use_batchnorm = config.get('use_batchnorm', use_batchnorm)
                self.hidden_channels = config.get('hidden_channels', hidden_channels)
            else:
                self.use_classifier = config.use_classifier
                self.latent_size = config.latent_size
                self.use_latent_split = config.use_latent_split
                self.dropout_rate = config.dropout_rate
                self.use_lstm = config.use_lstm
                self.use_residual = config.use_residual
                self.use_batchnorm = config.use_batchnorm
                self.hidden_channels = config.hidden_channels

        self.encoder = Encoder(
            latent_size=self.latent_size,
            use_lstm=self.use_lstm,
            hidden_channels = self.hidden_channels,
            use_residual = self.use_residual
        )

        self.decoder = Decoder(
            latent_size=self.latent_size,
            use_lstm=self.use_lstm,
            hidden_channels = self.hidden_channels,
            use_residual = self.use_residual
        )
        
        # reset this every instantiation, if we retrain later we don't want to affect this
        self.decay = 0 
        self.decay_rate = -1e-3
        self.decay_offset = 1000
        self.decayed = False


    def forward(self, x, return_all=False, hidden=None):
        # encode
        z_seq,residual = self.encoder(x)

        # residual logic, add noise to residual during training, and always scale the residual by the decay
        if self.use_residual and not self.decayed:
            if self.training:
                residual = residual + (0.8 * (torch.std(residual) + 1e-6) * torch.randn_like(residual, device=residual.device, dtype=residual.dtype))
            weight = F.sigmoid(torch.tensor(self.decay_rate * (self.decay -  self.decay_offset), device= x.device, dtype= x.dtype)) 
            decayed = F.relu(weight - 1e-3)
            self.decayed = decayed.cpu().item() == 0
            weight = weight * decayed * (1 / (torch.std(residual) + 1e-6)) # also normalize so that residual can't just try to get super large

            residual = residual * weight
            if(self.training):
                self.decay += 1
        # decode
        x_rec = self.decoder(z_seq, residual if (self.use_residual and not self.decayed) else torch.zeros_like(residual, device=residual.device, dtype=residual.dtype))


        return x_rec, z_seq

    def encode(self, x):
        z_seq, z_last = self.encoder(x)
        return z_seq, z_last

    def decode(self, z_seq):
        return self.decoder(z_seq)
