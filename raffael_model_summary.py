import torch
import torch.nn.functional as F
from torchinfo import summary
import sys
import os
from raffael_model import ConvLSTMAutoencoder
def main():
    model_lat = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=False,
        num_classes=2,
        use_latent_split=True,  # ENABLE LATENT SPLIT
        # Ablation parameters
        dropout_rate=0.1,
        use_convlstm=True,
        use_residual=True,
        use_batchnorm=True
    )

    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=False,
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=0.1,
        use_convlstm=True,
        use_residual=True,
        use_batchnorm=True
    )

    print("no split")
    summary(model, input_size = (1,1,1,128,128))
    print("split empty")
    summary(model_lat, input_size = (1,1,1,128,128), empty_well=True)
    print("split embryo")
    summary(model_lat, input_size = (1,1,1,128,128), empty_well=False)
if __name__ == "__main__":
    main()

