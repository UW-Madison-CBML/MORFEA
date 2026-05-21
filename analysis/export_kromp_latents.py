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
GRADES = ["A", "B", "C"] # I believe it is this order since 0 seems most prominent

def export_kromp(model):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False
    # -----------------------------------------------------
    # load the train and test csvs and rename accordingly
    silver_df = pd.read_csv(os.path.join("Blastocyst_Dataset", "Gardner_train_silver.csv"), delimiter=";", keep_default_na=True).rename(columns={"ICM_silver":"ICM", "TE_silver":"TE"})
    gold_df = pd.read_csv(os.path.join("Blastocyst_Dataset", "Gardner_test_gold_onlyGardnerScores.csv"), delimiter=";", keep_default_na=True).rename(columns={"ICM_gold":"ICM", "TE_gold":"TE"})

    # drop embryos with blastocoel development score < 2, such embryos are not developed enough to be graded for ICM and TE
    silver_df = silver_df[silver_df["EXP_silver"] >= 2].drop(columns=["EXP_silver"])
    gold_df = gold_df[gold_df["EXP_gold"] >= 2].drop(columns=["EXP_gold"])

    # concat and reset index
    metadata_df = pd.concat([gold_df, silver_df], axis=0, ignore_index=True) # axis 0 = along the index
    
    # drop NA
    metadata_df = metadata_df.dropna(subset=["TE","ICM"])
    
    # fix grades to be in a common format
    te_col = [GRADES[int(i)] for i in metadata_df["TE"].to_list()]
    icm_col = [GRADES[int(i)] for i in metadata_df["ICM"].to_list()]
    metadata_df["TE"] = te_col
    metadata_df["ICM"] = icm_col

    #--------------------------------------------------------------
    # now let's get the data set up
    
    print(metadata_df.head())
    image_abs_paths = [os.path.abspath(os.path.join("Blastocyst_Dataset", "Images", f"{img}")) for img in metadata_df["Image"].to_list() if os.path.exists(os.path.join("Blastocyst_Dataset", "Images", f"{img}"))]
    images_vol = np.stack([read_gray(path, 128) for path in image_abs_paths], axis=0)
    
    # for consistency normalize in the same way as the video latent exports
    images_vol = normalize_video(images_vol, "minmax01")
    
    images_tensor = torch.from_numpy(images_vol) # (B, 128, 128)
    images_tensor = images_tensor.unsqueeze(1).unsqueeze(1) # insert a channel and time dim of 1: (B, 1, 1, 128, 128)
    print("tensor shape: ", images_tensor.shape)
    # normal size of video tensors is (64, 32, 1 ...) so ~2300 should work as one batch
    
    # ----------------------------------------------------------- 
    # set up model
        
    model = model.to(DEVICE)
    model.eval()

    images_tensor = images_tensor.to(DEVICE)
    with torch.no_grad():
        _, latents = model(images_tensor)
    latents = latents.cpu().squeeze(1).numpy() # squeeze out time dim of 1: (B, 512)
    return metadata_df, latents 
        
        
def main(model_name):
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)
    
    metadata_df, latents = export_kromp(model)
    np.save(f"{model_name}.npy", latents)
    metadata_df.to_csv(f"{model_name}.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export frames from Kromp et al. dataset using a model on HF")

    parser.add_argument("--name", type=str, help="Name of the model", default="")

    args = parser.parse_args()


    main(args.name)
