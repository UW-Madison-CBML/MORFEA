# class for new ViT models (both with LSTM and 3d attention?)
import torch
import torchvision
import torch.nn.functional as F
class ViTEncoder(torch.nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None) # get the model
        # steal model components
        self.patch_embed = vit.conv_proj
        self.encoder = vit.encoder
        self.class_token = vit.class_token
        self.seq_length = vit.seq_length  

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.class_token.expand(x.shape[0], -1, -1)  
        x = torch.cat([cls, x], dim=1) 
        x = self.encoder(x)
        return x

class ViTLSTMAE(torch.nn.Module):
    def __init__(self, pretrained = True, latent_dim_per_token = 16, use_lstm = True):
        super().__init__()
        self.num_tokens = 196
        self.latent_dim_per_token = latent_dim_per_token
        self.use_lstm = use_lstm
        self.vit_enc = ViTEncoder(pretrained) # TODO this ViT is very small
        # remove the classification head just use transformer itself
        self.vit_enc.heads = torch.nn.Identity()
        self.lin1 = torch.nn.Linear(768, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.dropout = torch.nn.Dropout(0.2)
        if(self.use_lstm):
            self.lstm = torch.nn.LSTM(self.latent_dim_per_token * self.num_tokens, self.latent_dim_per_token * self.num_tokens, batch_first=True)
        self.positional_embedding = torch.nn.Embedding(self.num_tokens, 16)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=16, nhead=8)
        self.transformer_dec = torch.nn.TransformerDecoder(decoder_layer, num_layers=6) # TODO what norm to use here
        self.lin3 = torch.nn.Linear(16, 256) # patches of 16x16

        

    def forward(self, x):
        B,T,C,H,W =  x.shape # B, T, 1, 224,224
        x_flat_seq = x.view(B*T,1,224,224).repeat(1,3,1,1) # TODO don't do this
        transformed = self.vit_enc(x_flat_seq) # B*T, 197, 768
        print(transformed.shape)
        embeddings = transformed[:,1:,:] # B*T, 196, 768
        #num_tokens = embedding.shape[1]
        embeddings = F.relu(self.lin1(embeddings)) # B*T,196, 32
        embedding = self.dropout(embeddings) 
        embeddings = F.relu(self.lin2(embeddings)) # B*T,196, 16
        embeddings = embeddings.reshape(B,T, self.num_tokens * self.latent_dim_per_token) # unflatten the batch and sequence dims, and flatten the token and embedding dim
        if(self.use_lstm):
            latents, (_,_) = self.lstm(embeddings) # get the final LSTM sequence
        else:
            latents = embeddings
        latents_flat = latents.reshape(B*T, self.num_tokens, self.latent_dim_per_token) # reflatten batch and time dims, bring out token dim
        print(latents_flat.shape)
        latents_flat = latents_flat.permute(1,0,2) # transformer decoder needs batch second
        pos_enc = self.positional_embedding(torch.arange(self.num_tokens, device = x.device)[None,:]).repeat(B*T,1,1) # B*T, self.num_tokens, latent_dim_per_token
        pos_enc = pos_enc.permute(1,0,2) # need batch second
        x_rec = self.transformer_dec(pos_enc, latents_flat) # 196, B*T, 16
        x_rec = x_rec.permute(1, 0, 2) 
        x_rec = F.relu(self.lin3(x_rec)).reshape(B*T,14,14,16,16) # B*T, 196, 256
        x_rec = x_rec.permute(0, 1,3,2,4).reshape(B*T, 224, 224)
        x_rec = x_rec.reshape(B,T, 224,224)[:,:,None,:,:] # B,T, 1, 224,224
        return x_rec, latents
        
class SmallViTLSTMAE(torch.nn.Module):
    def __init__(self, pretrained = True, latent_dim=512, use_lstm = True):
        super().__init__()
        self.num_tokens = 196
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.vit_enc = ViTEncoder(pretrained) # TODO this ViT is very small
        # remove the classification head just use transformer itself
        self.vit_enc.heads = torch.nn.Identity()
        self.lin1 = torch.nn.Linear(768, 512)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(512, self.latent_dim)
        if(self.use_lstm):
            self.lstm = torch.nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        self.positional_embedding = torch.nn.Embedding(self.num_tokens, self.latent_dim)
        self.decoder_up = torch.nn.Linear(self.latent_dim, self.latent_dim)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=8)
        self.transformer_dec = torch.nn.TransformerDecoder(decoder_layer, num_layers=6) # TODO what norm to use here
        self.lin3 = torch.nn.Linear(self.latent_dim, 256) # patches of 16x16

        

    def forward(self, x):
        B,T, C,H,W =  x.shape # B, T, 1, 224,224
        x_flat_seq = x.reshape(B*T,1,224,224).expand(-1,3,-1,-1).contiguous() # TODO don't do this
        transformed = self.vit_enc(x_flat_seq) # B*T, 197, 768
        embeddings = transformed[:,0,:] # B*T, 768
        embeddings = F.relu(self.lin1(embeddings)) # B*T, 512
        embeddings = self.dropout(embeddings) 
        embeddings = F.relu(self.lin2(embeddings)) # B*T, 512
        embeddings = embeddings.reshape(B,T, self.latent_dim) # unflatten the batch and sequence dims, and flatten the token and embedding dim
        if(self.use_lstm):
            latents, (_,_) = self.lstm(embeddings) # get the final LSTM sequence
        else:
            latents = embeddings
        latents_flat = latents.reshape(B*T, self.latent_dim) # reflatten batch and time dims, bring out token dim
        print(latents_flat.shape)
        #padding = torch.zeros((self.num_tokens, B*T, self.latent_dim), device = x.device) 
        #latents_flat = F.relu(self.decoder_up(latents_flat))
        tokens = latents_flat[None,:,:].contiguous() # 1, B*T, 256, all 196 positional tokens with cross attend with this
        pos_enc = self.positional_embedding(torch.arange(self.num_tokens, device = x.device)[:,None]).expand(-1,B*T,-1).contiguous() # B*T, self.num_tokens + 1, latent_dim
        x_rec = self.transformer_dec(pos_enc, tokens) # 196, B*T, self.latent_dim
        x_rec = x_rec.permute(1, 0, 2) # B*T, 196, self.latent_dim
        x_rec = F.relu(self.lin3(x_rec)) # B*T, 196, 256
        x_rec = x_rec.reshape(B*T, 14,14, 16,16)
        x_rec = x_rec.permute(0, 1, 3, 2, 4) # B*T, 14, 16, 14, 16
        x_rec = x_rec.reshape(B*T, 224, 224)
        x_rec = x_rec.reshape(B,T, 224,224)[:,:,None,:,:] # B,T, 1, 224,224
        return x_rec, latents
      
if __name__ == "__main__":
    model = ViTLSTMAE()
    image = torch.randn((1,1,1,224,224))
    recon, latent = model(image)
    
