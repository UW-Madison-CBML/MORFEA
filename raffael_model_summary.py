import torch
import torch.nn.functional as F
from torchinfo import summary
import sys
import os;
from raffael_model import Encoder, Decoder
def main():
    enc = Encoder()
    dec = Decoder(1)

    summary(enc, input_size = (1,1,1,128,128))
    summary(dec, imput_size = (1,1, 256, 16, 16))
if __name__ == "__main__":
    main()

