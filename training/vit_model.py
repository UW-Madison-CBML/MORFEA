# class for new ViT models (both with LSTM and 3d attention?)
import torch
import torchvision
class ViTLSTMAE(torch.nn.Module):
    def __init__(self, pretrained = False, latent_dim_per_token = 16, use_lstm = True):
        super().__init__()
        self.latent_dim_dim_per_token = latent_dim_per_token
        self.use_lstm = use_lstm
        self.vit_enc = torchvision.models.vit_b_16(weights='DEFAULT' if pretrained else None) # TODO this ViT is very small
        self.lin1 = self.linear(768, 32)
        self.lin2 = self.linear(32, 16)
        self.dropout = torch.nn.Dropout(0.2)
        if(self.use_lstm):
            self.lstm = torch.nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        self.positional_embedding = torch.nn.Embedding()
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=16, nhead=8)
        self.vit_dec = torch.nn.TransformerDecoder(decoder_layer, num_layers=6, norm=torch.nn.BatchNorm()) # TODO what norm to use here

        

    def forward(self, x):
        B,T,C,H,W =  x.shape # B, T, 1, 224,224
        x_flat_seq = x.view(B*T,1,224,224) 
        transformed = vit_enc(x) # B*T, 197, 768
        embeddings = transformed[:,1:,:] # B*T, 196, 768
        num_tokens = embedding.shape[1]
        embeddings = F.relu(self.lin1(embeddings)) # B*T,196, 32
        
        embeddings = F.relu(self.lin2(embeddings)) # B*T,196, 16
        embeddings = embeddings.reshape(B,T,num_tokens * self.latent_dim_per_token) # unflatten the batch and sequence dims
        if(self.use_lstm):
            latents, (_,_) = self.lstm(embeddings) # get the final LSTM sequence
        else:
            latents = embeddings
        latents_flat = latents.reshape(B*T, num_tokens, self.latent_dim_per_token) # 
        x = vit_dec(latents_flat)
        return x, latents
        
