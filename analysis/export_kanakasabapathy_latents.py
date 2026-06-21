import pandas as pd
import numpy as np
import torch
from ae_model import ConvLSTMAutoencoder
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from dataset_ivf_embryo import IVFEmbryoDataset
from ae_model import ConvLSTMAutoencoder
from huggingface_hub import login, HfApi
from dataset_ivf_embryo import read_gray, normalize_video
import os
import gc
GRADES = ["A", "B", "C"] # I believe it is this order since 0 seems most prominent

def export_kanakasabapathy(model, image_size = 128):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    images_3 = [os.path.join("kanakasabapathy","3",path) for path in os.listdir(os.path.join("kanakasabapathy","3"))]
    images_4 = [os.path.join("kanakasabapathy","4",path) for path in os.listdir(os.path.join("kanakasabapathy","4"))]
    images_5 = [os.path.join("kanakasabapathy","5",path) for path in os.listdir(os.path.join("kanakasabapathy","5"))]
    paths = images_3 + images_4 + images_5
    grades = (["C"] * len(images_3))+ (["B"] * len(images_4))+(["A"] * len(images_5))
    metadata_df = pd.DataFrame({"path":paths, "TE":grades, "embryo_id":np.arange(len(paths))}) #spoof the embryo id as just a number

    
    #--------------------------------------------------------------
    # now let's get the data set up
    
    image_abs_paths = paths
    images_vol = np.stack([read_gray(path, image_size, 0) for path in image_abs_paths], axis=0)
    
    # for consistency normalize in the same way as the video latent exports
    images_vol = normalize_video(images_vol, "minmax01")
    
    images_tensor = torch.from_numpy(images_vol) # (B, 128, 128)
    images_tensor = images_tensor.unsqueeze(1) # insert a channel and time dim of 1: (B, 1, 1, 128, 128)
    assert len(images_tensor.shape) == 5, f"expected 5 dim tensor, got {len(images_tensor.shape)}"
    # normal size of video tensors is (64, 32, 1 ...) so ~2300 should work as one batch
    
    # ----------------------------------------------------------- 
    # set up model
        
    model = model.to(DEVICE)
    model.eval()

    images_tensor = images_tensor.to(DEVICE)
    with torch.no_grad():
        imgs, latents = model(images_tensor)
    latents = latents.cpu().squeeze(1).numpy() # squeeze out time dim of 1: (B, 512)
    imgs = imgs.cpu().squeeze(1).squeeze(1).numpy() # (B, 128, 128)
    gc.collect()
    torch.cuda.memory.empty_cache()
    model.train()
    return metadata_df, latents, imgs
        
        
def main(model_name):
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)
    
    metadata_df, latents, _ = export_kanakasabapathy(model)
    np.save(f"{model_name}.npy", latents)
    metadata_df.to_csv(f"{model_name}.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export frames from Kanakasabapathy et al. dataset using a model on HF")

    parser.add_argument("--name", type=str, help="Name of the model", default="")

    args = parser.parse_args()


    main(args.name)
