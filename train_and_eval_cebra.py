import cebra
import torch
from cebra import CEBRA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from raffael_model import ConvLSTMAutoencoder
def main(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    login(os.getenv("HF_KEY"))
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/" + model_name)     
    model.to(device)
    VAL_EMBRYOS = pd.read_csv("embryo_dataset_grades.csv").rename(columns={"video_name":"embryo_id"}).dropna(subset=["ICM"])["embryo_id"].astype(str).tolist()
    grades_df = pd.read_csv(os.path.abspath("embryo_dataset_grades.csv")).rename(columns={"video_name":"embryo_id"}).dropna(subset=["TE"])[["embryo_id","TE"]]

    full_seq_df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})    
    full_seq_df = full_seq_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")
    full_seq_val_mask = full_seq_df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    full_seq_df_val = full_seq_df[full_seq_val_mask] # just look at validation ICM embryos
    
    full_seq_df_train = full_seq_df #[~full_seq_val_mask] train on whole dataset # just look at validation ICM embryos
    full_seq_dataset_train = IVFEmbryoDataset(full_seq_df_train, resize=128, norm="minmax01")

    
    full_seq_loader_train = DataLoader(
        full_seq_dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False 
    )
 
    cebra_time_model = CEBRA(model_architecture="offset10-model",
                        batch_size=512,
                        learning_rate=1e-2,
                        temperature=1.0,
                        output_dimension=3,
                        num_hidden_units=128,
                        max_iterations=5000,
                        distance="euclidean",
                        conditional="time",
                        device="cuda_if_available",
                        verbose=True,
                        time_offsets=10)
    print(model_name) 
    cebra_latents = []
    cebra_labels = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for embryo_vol in full_seq_loader_train:
            embryo_vol = embryo_vol.to(device)
            _, z_seq = model(embryo_vol)
            traj = z_seq.cpu().detach().numpy()[0] # batch size one just use that batch
            cebra_latents.append(traj)
            cebra_labels.append((np.arange(len(traj)) + offset).reshape(-1, 1).astype(np.float32))
            offset += len(traj) + 10000
    import gc
    del embryo_vol, z_seq 
    gc.collect()
    torch.cuda.empty_cache()
    cebra_time_model.fit(np.concatenate(cebra_latents, axis=0), np.concatenate(cebra_labels, axis=0))
    cebra_time_model.save(f"{model_name}_cebra_time_model.pt")

    torch.cuda.empty_cache()
    
    max_imgs = 16
    with torch.no_grad():
        for grade in ["A", "B", "C"]:
            grade_ds = IVFEmbryoDataset(full_seq_df[full_seq_df["TE"] == grade], resize=128, norm="minmax01")
            grade_loader = DataLoader(
                grade_ds,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False 
            )
            print(len(grade_loader))
            fig, axes = plt.subplots(4, 4, figsize=(20, 20),subplot_kw={'projection': '3d'})
            axes = axes.ravel()
            d3_trajs = []
            for j, embryo_vol in enumerate(grade_loader):
                if(j >= len(axes)):
                    break
                embryo_vol = embryo_vol.to(device) 
                _, z_seq = model(embryo_vol)
                traj = z_seq.cpu().detach().numpy()[0] # batch size one just use that batch
                cebra_embedding = cebra_time_model.transform(traj) # i guess dont batch it?
                d3_trajs.append(cebra_embedding)
            del embryo_vol, z_seq
            x_list = [x for t in d3_trajs for x in t[:, 0]]
            y_list = [y for t in d3_trajs for y in t[:, 1]]
            z_list = [z for t in d3_trajs for z in t[:, 2]]
            x_lim = (min(x_list),max(x_list))
            y_lim = (min(y_list),max(y_list))
            z_lim = (min(z_list),max(z_list))

            im = None
            for i, d3_traj in enumerate(d3_trajs):
                ax = axes[i]
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=np.linspace(0,1,d3_traj.shape[0]), cmap='viridis')
                
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim3d(z_lim) 
                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Time')
 
            fig.savefig(os.path.join("cebra_plots",f"{grade}.png"))
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': '3d'})
            im = None
            for i, d3_traj in enumerate(d3_trajs):
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=np.linspace(0,1,d3_traj.shape[0]), cmap='viridis')
                
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim3d(z_lim) 
            ax.set_xlabel("Cebra 1")
            ax.set_ylabel("Cebra 2")
            ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Time')
            fig.savefig(os.path.join("cebra_plots",f"grouped_{grade}.png"))
            plt.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
