import cebra
import torch
from cebra import CEBRA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huggingface_hub import login
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from raffael_model import ConvLSTMAutoencoder
def main(model_name):
    
    login(os.getenv("HF_KEY"))
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/" + model_name)     
    VAL_EMBRYOS = pd.read_csv("embryo_dataset_grades.csv").rename(columns={"video_name":"embryo_id"}).dropna(subset=["ICM"])["embryo_id"].astype(str).tolist()
    full_seq_df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})
    full_seq_val_mask = full_seq_df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    full_seq_df_val = full_seq_df[full_seq_val_mask] # just look at validation ICM embryos
    
    full_seq_df_train = full_seq_df[~full_seq_val_mask] # just look at validation ICM embryos
    full_seq_dataset_val = IVFEmbryoDataset(full_seq_df_val, resize=128, norm="minmax01")
    full_seq_dataset_train = IVFEmbryoDataset(full_seq_df_train, resize=128, norm="minmax01")

    full_seq_loader_val = DataLoader(
        full_seq_dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False 
    )
    full_seq_loader_train = DataLoader(
        full_seq_dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False 
    )
 
    cebra_time_model = CEBRA(model_architecture="offset10-model",
                        batch_size=128,
                        learning_rate=1e-2,
                        temperature=1,
                        output_dimension=3,
                        num_hidden_units=128,
                        max_iterations=3000,
                        distance="euclidean",
                        conditional="time",
                        device="cuda_if_available",
                        verbose=True,
                        time_offsets=10)
 
    cebra_latents = []
    cebra_labels = []
    model.eval()
    with torch.no_grad():
        for embryo_vol in full_seq_loader_train:
            embryo_vol = embryo_vol.to(DEVICE)
            _, z_seq = model(embryo_vol)
            traj = z_seq.cpu().detach().numpy()[0] # batch size one just use that batch
            cebra_latents.append(traj)
            cebra_labels.append(np.arange(len(traj)).reshape(-1, 1).astype(np.float32))
    cebra_time_model.fit(cebra_latents, cebra_labels)
    #cebra_time_model.save("cebra_time_model.pt")


    rand_img = np.random.randint(40, 64) 
    with torch.no_grad():
        for i, embryo_vol in enumerate(full_seq_loader_val):
            row = full_seq_df_val.iloc[i] 
            embryo_vol = embryo_vol.to(DEVICE) 
            _, z_seq = model(embryo_vol)
            traj = z_seq.cpu().detach().numpy()[0] # batch size one just use that batch
                            
            cebra_embedding = cebra_time_model.transform(traj, session_id=0) # i guess dont batch it?
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            im = ax.scatter(cebra_embedding[:,0], cebra_embedding[:,1], cebra_embedding[:,2], c=np.linspace(0,1,cebra_embedding.shape[0]), cmap='viridis')

            ax.set_xlabel("Cebra 1")
            ax.set_ylabel("Cebra 2")
            ax.set_zlabel("Cebra 3")
            
            plt.colorbar(im, ax=ax)
            
            fig.savefig(os.path.join("cebra_plots",f"{row['embryo_id']}.png"))
            plt.close()
if __name__ == "__main__":
    import sys
    main(sys.argv[1])
