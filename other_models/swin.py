import requests
import torch
from PIL import Image

import timm

def main():
    # Hyperparameters
    lr = 3e-4
    batch_size = 64
    
    # download model with timm
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=18)
    model.to(DEVICE)
    
    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        outputs = model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    

if __name__ == "__main__":
    main()
