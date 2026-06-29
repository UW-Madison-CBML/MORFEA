# class for new ViT models (both with LSTM and 3d attention?)
import torch
import torchvision
import torch.nn.functional as F
from ae_model import Decoder

from huggingface_hub import PyTorchModelHubMixin
class ViTEncoder(torch.nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        vit = torchvision.models.vit_l_16(weights='IMAGENET1K_V1' if pretrained else None) # get the model
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
        self.positional_embedding = torch.nn.Embedding(self.num_tokens, self.latent_dim_per_token)
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
        latents_flat = latents_flat.permute(1,0,2).contiguous() # transformer decoder needs batch second
        pos_enc = self.positional_embedding(torch.arange(self.num_tokens, device = x.device)[:, None]).repeat(1, B*T, 1) # B*T, self.num_tokens, latent_dim_per_token
        x_rec = self.transformer_dec(pos_enc, latents_flat) # 196, B*T, 16
        x_rec = x_rec.permute(1, 0, 2).contiguous()
        x_rec = F.sigmoid(self.lin3(x_rec)).reshape(B*T,14,14,16,16) # B*T, 196, 256
        x_rec = x_rec.permute(0, 1,3,2,4).contiguous().reshape(B*T, 224, 224)
        x_rec = x_rec.reshape(B,T, 224,224)[:,:,None,:,:] # B,T, 1, 224,224
        return x_rec, latents
        
class SmallViTLSTMAE(torch.nn.Module):
    def __init__(self, pretrained = True, latent_dim=1024, use_lstm = True):
        super().__init__()
        self.num_tokens = 196
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.vit_enc = ViTEncoder(pretrained) # TODO this ViT is very small
        self.lin1 = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(1024, self.latent_dim)
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

        #padding = torch.zeros((self.num_tokens, B*T, self.latent_dim), device = x.device) 
        #latents_flat = F.relu(self.decoder_up(latents_flat))
        tokens = latents_flat[None,:,:].contiguous() # 1, B*T, self.latent_dim, all 196 positional tokens with cross attend with this
        pos_enc = self.positional_embedding(torch.arange(self.num_tokens, device = x.device)[:,None]).expand(-1,B*T,-1).contiguous() # self.num_tokens, B*T, latent_dim
        x_rec = self.transformer_dec(pos_enc, tokens) # 196, B*T, self.latent_dim
        x_rec = x_rec.permute(1, 0, 2).contiguous() # B*T, 196, self.latent_dim
        x_rec = F.sigmoid(self.lin3(x_rec)) # B*T, 196, 256
        x_rec = x_rec.reshape(B*T, 14,14, 16,16)
        x_rec = x_rec.permute(0, 1, 3, 2, 4).contiguous() # B*T, 14, 16, 14, 16
        x_rec = x_rec.reshape(B*T, 224, 224)[:,None,:,:]
        x_rec = x_rec.reshape(B,T,C,H,W)
        return x_rec, latents

class ConvViTLSTMAE(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, config=None, pretrained = True, latent_dim=1024, use_lstm = True):
        super().__init__()
        # don't need either of these in config neither contains important information about the model
        self.pretrained=pretrained
        self.num_tokens = 196
        # these are relevant for model loading since they have to do with the model's shape
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        #if(config is not None):
        if config is not None:
            if isinstance(config, dict):
                self.latent_dim = config.get('latent_dim', latent_dim)
                self.use_lstm = config.get('use_lstm', use_lstm)
            else:
                self.latent_dim = config.latent_dim
                self.use_lstm = config.use_lstm

        self.vit_enc = ViTEncoder(self.pretrained) # TODO this ViT is very small
        self.lin1 = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(768, self.latent_dim)
        if(self.use_lstm):
            self.lstm = torch.nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        self.decoder = Decoder(latent_size = self.latent_dim, initial_resolution = 28, final_size=224, use_lstm = self.use_lstm) # 28 = 224/8
                

    def forward(self, x):
        B,T, C,H,W = x.shape # B, T, 1, 224,224
        assert len(x.shape) == 5, f"expected 5 dims, got {x.shape}"
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

        x_rec = self.decoder(latents) # B,T,L
        return x_rec, latents

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dim)
        )

    def forward(self, x):
        B,T,P,L =  x.shape
        x = x.view(B*T,P,L)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.view(B,T,P,L)
        return x

class ViTMAE(torch.nn.Module):
    def mask(self, x):
        """
        x: the patches pre masking
        returns:
            masked_patches: B,T,num_unmasked, self.patch_size ** 2; the masked patches without padding in between
            mask: B,num_patches; type torch.bool; the boolean mask representing the patches that are unmasked (True) across batch dim (same across time dim) 
            mask_indices: B,num_patches; type=torch.uint; the padded indices that index masked_patches telling where to gather them from in each sequence in the batch
            pos_indices: B, num_unmasked; type=torch.uint; the indices of each unmasked patch indexing in x, for use with positional encoding
        """
        B,T,P,L = x.shape
        noise = torch.rand(B,P, device=x.device)
        rand_indices = noise.argsort(dim=1)
        pos_indices = rand_indices[:,:self.num_unmasked].sort(dim=1).values
        int_mask = F.one_hot(pos_indices, num_classes=self.num_patches).sum(dim=1)
        assert ((int_mask == 0) | (int_mask == 1)).all(), "error somehow pos_indices is not unique"
        mask = int_mask.to(torch.bool) 

        mask_indices = torch.where(mask, mask.cumsum(dim=1) - 1, torch.tensor(-1,device=x.device, dtype=torch.int)) # cumsum is too high by 1
        return torch.gather(x, 2, pos_indices[:,None,:, None].repeat(1,T,1,L)), mask, mask_indices, pos_indices


    def positional_encoding(self,shape,device,indices=None):
        B,T,P,L = shape # B,T,P,128
        assert P == self.num_patches, f"expected pos_enc shape[2] to be {self.num_patches}, got {P}"
        pos_enc = self.embedding(torch.arange(P,device=device)) # P, 128
        pos_enc = pos_enc[None,None,:,:].repeat(B,T,1,1)
        if indices is not None:
            assert indices.shape == (B,self.num_unmasked), f"expected indices.shape = [{B},{self.num_unmasked}], got {indices.shape}" 
            assert 0 <= indices.min()  and indices.max() < self.num_patches, f"expected indices to be in range [0,{self.num_patches}), got [{indices.min()},{indices.max()})"
            return torch.gather(pos_enc, 2, indices[:,None,:,None].repeat(1,T,1,128)) # last dim is embedding dim 128
        else:
            return pos_enc

    def patchify(self, x):
        B,T,C,H,W = x.shape
        assert C == 1, f"expected 1 channel got {C}"
        assert H == W and H == self.image_size, f"expected image size [{self.image_size}, {self.image_size}], got [{H},{W}]"
        x = x.squeeze(2)
        x = x.reshape(B,T,self.image_size//self.patch_size,self.patch_size, self.image_size//self.patch_size,self.patch_size)
        x = x.permute(0,1,2,4,3,5).contiguous()
        return x.reshape(B, T, (self.image_size // self.patch_size)**2, self.patch_size**2)

    def unpatchify(self, x):
        B,T,P,L = x.shape
        assert P == (self.image_size // self.patch_size) ** 2 and L == self.patch_size ** 2, f"expected image size [{B},{T},{self.P == (self.image_size // self.patch_size) ** 2}, {L == self.patch_size ** 2}], got [{B},{T},{P},{L}]"
        x = x.reshape(B,T,self.image_size // self.patch_size,self.image_size // self.patch_size,self.patch_size,self.patch_size)
        x = x.permute(0,1,2,4,3,5).contiguous()
        x = x.reshape(B,T,self.image_size, self.image_size)
        return x.unsqueeze(2)


    
    def __init__(self, image_size=224, patch_size=16, latent_dim_per_token=128, num_unmasked=80, num_blocks=8):
        super().__init__()
        self.image_size = image_size 
        self.latent_dim_per_token = latent_dim_per_token
        assert image_size % patch_size == 0, f"expected patch_size: {patch_size}, to divide image_size: {image_size}"
        self.patch_size = patch_size
        assert 0 <= num_unmasked and num_unmasked <= (self.image_size // self.patch_size) ** 2, f"expected num_unmasked: {num_unmasked} to be less than num_patches: {(self.image_size // self.patch_size) ** 2}"
        self.num_blocks = num_blocks
        self.num_unmasked = num_unmasked
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.embedding = torch.nn.Embedding(self.num_patches, 128)
        self.transformer_encoder = torch.nn.ModuleList([TransformerBlock(dim=128,num_heads=8) for i in range(self.num_blocks)])
        self.transformer_decoder = torch.nn.ModuleList([TransformerBlock(dim=128,num_heads=8) for i in range(self.num_blocks)])

        self.lin1 = torch.nn.Linear(self.patch_size**2,128)
        self.lin2 = torch.nn.Linear(128,self.latent_dim_per_token)

        self.lin3 = torch.nn.Linear(self.latent_dim_per_token, self.latent_dim_per_token)

        self.lin4 = torch.nn.Linear(self.latent_dim_per_token, 128)
        self.lin5 = torch.nn.Linear(128, self.patch_size**2)

    def masked_loss(self, target, pred, mask):
        """
        target: B,T,P,L; patched target
        pred: B,T,P,L; patched prediction / recon 
        """
        assert target.shape == pred.shape, f"expected target shape to match pred shape, got pred: {pred.shape}, target: {target.shape}"
        assert mask.shape == (pred.shape[0],pred.shape[2]), f"expected mask.shape = [{pred.shape[0]}, {pred.shape[2]}], got {mask.shape}"
        mask = mask[:,None,:].repeat(1, target.shape[1], 1)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # B,T,P
        loss = (loss * (~ mask)).sum() / (~ mask).sum() # 1 in mask means patch is unmasked, so we need to compare masked pixels cause those are actually reconstructed
        return loss     
          
    def forward(self, x):
        B,T,C,H,W = x.shape
        patches = self.patchify(x) # B,T,num_patches, self.patch_size ** 2
        masked_patches, mask, mask_indices, pos_indices = self.mask(patches) # B,T,num_unmasked, patch_size**2; B, num_patches-torch.bool; B, num_unmasked-torch.int
        
        pos_enc = self.positional_encoding(patches.shape, x.device, indices=pos_indices)
        masked_patches = F.relu(self.lin1(masked_patches))
        masked_patches = masked_patches + pos_enc
        for enc in self.transformer_encoder:
            masked_patches = enc(masked_patches) # B, T, num_unmasked, patch_size**2
        latent_patches = self.lin2(masked_patches)
        latents = latent_patches.reshape(B,T,self.num_unmasked * self.latent_dim_per_token)
        latent_patches = F.relu(self.lin3(latent_patches))
        latent_patches = F.relu(self.lin4(latent_patches))
        latent_patches_pad_0 = torch.cat([torch.zeros((B,T,1,self.latent_dim_per_token), device=latent_patches.device),latent_patches],dim=2)
        padded_patches = torch.gather(latent_patches_pad_0, 2, (mask_indices[:,None,:,None]+1).repeat(1,T,1,self.latent_dim_per_token)) # need to add 1 to account for offset
        dec_pos_enc = self.positional_encoding(padded_patches.shape, x.device) 
        patches_to_decode = padded_patches + dec_pos_enc 
        for dec in self.transformer_decoder:
            patches_to_decode = dec(patches_to_decode) 
        recon_patches = F.sigmoid(self.lin5(patches_to_decode))
        x_recon = self.unpatchify(recon_patches)
        return x_recon, latents, mask, self.masked_loss(patches, recon_patches, mask)
        
        

if __name__ == "__main__":
    model = ViTMAE()
    image = torch.randn((1,1,1,224,224))
    print(model(image))
    
