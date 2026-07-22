import pandas as pd
import numpy as np
import torch
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from dataset_ivf_embryo import IVFEmbryoDataset
from ae_model import ConvGRUAutoencoder
from huggingface_hub import login, HfApi
from dataset_ivf_embryo import read_gray, normalize_video
import os
import gc
GRADES = ["A", "B", "C"] # I believe it is this order since 0 seems most prominent
# binary_classification: classify by blastocyst non blastocyst (1,2 vs 3,4,5) if True classify quality (3 vs 4 vs 5) if False
def export_kanakasabapathy(model, image_size = 128, vitmae=False, binary_classification=True):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if binary_classification:
        images_12 = [os.path.join("kanakasabapathy","1",path) for path in os.listdir(os.path.join("kanakasabapathy","1"))] + [os.path.join("kanakasabapathy","2",path) for path in os.listdir(os.path.join("kanakasabapathy","2"))]
        images_345 = [os.path.join("kanakasabapathy","3",path) for path in os.listdir(os.path.join("kanakasabapathy","3"))] + [os.path.join("kanakasabapathy","4",path) for path in os.listdir(os.path.join("kanakasabapathy","4"))] + [os.path.join("kanakasabapathy","5",path) for path in os.listdir(os.path.join("kanakasabapathy","5"))]
        paths = images_12 + images_345
        grades = (["C"] * len(images_12)) + (["A"] * len(images_345))
        metadata_df = pd.DataFrame({"path":paths, "TE":grades, "embryo_id":np.arange(len(paths))}) #spoof the embryo id as just a number


    else:
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
    images_tensor = images_tensor.unsqueeze(1).unsqueeze(1) # insert a channel and time dim of 1: (B, 1, 1, 128, 128)
    assert len(images_tensor.shape) == 5, f"expected 5 dim tensor, got {len(images_tensor.shape)}"
    # normal size of video tensors is (64, 32, 1 ...) so ~2300 should work as one batch
    
    # ----------------------------------------------------------- 
    # set up model
        
    model = model.to(DEVICE)
    model.eval()
    images_tensor1 = images_tensor[:images_tensor.shape[0]//2]
    images_tensor2 = images_tensor[images_tensor.shape[0]//2:]
    images_tensor1 = images_tensor1.to(DEVICE)
    images_tensor2 = images_tensor2.to(DEVICE)
    with torch.no_grad():
        if(vitmae):
            imgs1, latents1,_,_ = model(images_tensor1)
            imgs2, latents2,_,_ = model(images_tensor2)
            imgs = torch.cat([imgs1, imgs2],dim=0)
            latents = torch.cat([latents1, latents2],dim=0)
        else:
            imgs1, latents1 = model(images_tensor1)
            imgs2, latents2 = model(images_tensor2)
            imgs = torch.cat([imgs1, imgs2],dim=0)
            latents = torch.cat([latents1, latents2],dim=0)
            
    latents = latents.cpu().squeeze(1).numpy() # squeeze out time dim of 1: (B, 512)
    imgs = imgs.cpu().squeeze(1).squeeze(1).numpy() # (B, 128, 128)
    gc.collect()
    torch.cuda.memory.empty_cache()
    model.train()
    return metadata_df, latents, imgs
        
        
def main(model_name):
    model = ConvGRUAutoencoder.from_pretrained("JensLundsgaard/"+model_name)
    
    metadata_df, latents, _ = export_kanakasabapathy(model, binary_classification=False)
    np.save(f"{model_name}.npy", latents)
    metadata_df.to_csv(f"{model_name}.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export frames from Kanakasabapathy et al. dataset using a model on HF")

    parser.add_argument("--name", type=str, help="Name of the model", default="")

    args = parser.parse_args()


    main(args.name)
