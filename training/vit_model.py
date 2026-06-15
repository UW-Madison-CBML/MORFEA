# class for new ViT models (both with LSTM and 3d attention?)
import torch
import torchvision
import torch.nn.functional as F
class ViTLSTMAE(torch.nn.Module):
    def __init__(self, pretrained = False, latent_dim_per_token = 16, use_lstm = True):
        super().__init__()
        self.num_tokens = 196
        self.latent_dim_dim_per_token = latent_dim_per_token
        self.use_lstm = use_lstm
        self.vit_enc = torchvision.models.vit_b_16(weights='DEFAULT' if pretrained else None) # TODO this ViT is very small
        self.lin1 = torch.nn.Linear(768, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.dropout = torch.nn.Dropout(0.2)
        if(self.use_lstm):
            self.lstm = torch.nn.LSTM(self.latent_dim_per_token * self.num_tokens, self.latent_dim_per_token * self.num_tokens, batch_first=True)
        self.positional_embedding = torch.nn.Embedding(self.num_tokens, 16)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=16, nhead=8)
        self.transformer_dec = torch.nn.TransformerDecoder(decoder_layer, num_layers=6, norm=torch.nn.BatchNorm()) # TODO what norm to use here
        self.lin3 = torch.nn.Linear(16, 256)# patches of 16x16

        

    def forward(self, x):
        B,T,C,H,W =  x.shape # B, T, 1, 224,224
        x_flat_seq = x.view(B*T,1,224,224) 
        transformed = self.vit_enc(x) # B*T, 197, 768
        embeddings = transformed[:,1:,:] # B*T, 196, 768
        #num_tokens = embedding.shape[1]
        embeddings = F.relu(self.lin1(embeddings)) # B*T,196, 32
        
        embeddings = F.relu(self.lin2(embeddings)) # B*T,196, 16
        embeddings = embeddings.reshape(B,T, self.num_tokens * self.latent_dim_per_token) # unflatten the batch and sequence dims, and flatten the token and embedding dim
        if(self.use_lstm):
            latents, (_,_) = self.lstm(embeddings) # get the final LSTM sequence
        else:
            latents = embeddings
        latents_flat = latents.reshape(B*T, self.num_tokens, self.latent_dim_per_token) # reflatten batch and time dims, bring out token dim
        latents_flat = latents.permute(1,0,2) # transformer decoder needs batch second
        pos_enc = self.positional_embedding(torch.arange(self.num_tokens, device = x.device)[None,:]).repeat(B*T,1) # B*T, self.num_tokens, latent_dim_per_token
        pos_enc = pos_enc.permute(1,0,2) # need batch first
        x_rec = self.transformer_dec(pos_enc, latents_flat) # 196, B*T, 16
        x_rec = x_rec.permute(1, 0, 2) 
        x_rec = F.relu(self.lin3(x_rec)).reshape(B*T,14,14,16,16) # B*T, 196, 256
        x_rec = x_rec.permute(0, 1,3,2,4).reshape(B*T, 224, 224)
        x_rec = x_rec.reshape(B,T, 224,224)[:,:,None,:,:] # B,T, 1, 224,224
        return x_rec, latents
        
if __name__ == "__main__":
    model = ViTLSTMAE()
    image = torch.randn((1,1,1,224,224))
    recon, latent = model(image)
