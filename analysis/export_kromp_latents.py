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
def main(model_name):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # -----------------------------------------------------
    # load the train and test csvs and rename accordingly
    silver_df = pd.read_csv(os.path.join("Blastocyst_Dataset", "Gardner_train_silver.csv"), delimiter=";").rename(columns={"ICM_silver":"ICM", "TE_silver":"TE"})
    gold_df = pd.read_csv(os.path.join("Blastocyst_Dataset", "Gardner_test_gold_onlyGardnerScores.csv"), delimiter=";").rename(columns={"ICM_gold":"ICM", "TE_gold":"TE"})

    # drop embryos with blastocoel development score < 2, such embryos are not developed enough to be graded for ICM and TE
    silver_df = silver_df[silver_df["EXP_silver"] < 2].drop(columns=["EXP_silver"])
    gold_df = gold_df[gold_df["EXP_gold"] < 2].drop(columns=["EXP_gold"])

    # concat and reset index
    metadata_df = pd.concat([gold_df, silver_df], axis=0, ignore_index=True) # axis 0 = along the index
    
    # fix grades to be in a common format
    TE = [GRADES[i] for i in metadata_df["TE"].to_list()]
    ICM = [GRADES[i] for i in metadata_df["ICM"].to_list()]
    metadata_df["TE"] = TE
    metadata_df["ICM"] = ICM

    #--------------------------------------------------------------
    # now let's get the data set up

    image_abs_paths = [os.path.abspath(os.path.join("Blastocyst_Dataset", "Images", f"{img}.png")) for img in metadata["Image"].to_list()]
    images_vol = np.stack([read_gray(path, 128) for path in image_abs_paths], axis=0)
    
    # for consistency normalize in the same way as the video latent exports
    images_vol = normalize_video(images_vol, "minmax01")
    
    images_tensor = torch.from_numpy(images_vol) # (B, 128, 128)
    images_tensor = images_tensor.unsqueeze(1).unsqueeze(1) # insert a channel and time dim of 1: (B, 1, 1, 128, 128)
    # normal size of video tensors is (64, 32, 1 ...) so ~2300 should work as one batch
    
    # ----------------------------------------------------------- 
    # load model
        
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)
    model = model.to(DEVICE)
    model.eval()

    images_tensor = images_tensor.to(DEVICE)
    with torch.no_grad():
        _, latents = model(images_tensor)
    latents = latents.cpu().squeeze(1).numpy() # squeeze out time dim of 1: (B, 512)
    
    np.save(latents, f"{model_name}.npy")
    metadata_df.to_csv(f"{model_name}.csv")
    
        
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export frames from Kromp et al. dataset using a model on HF")

    parser.add_argument("--name", type=str, help="Name of the model", default="")

    args = parser.parse_args()


    main(args.name)
