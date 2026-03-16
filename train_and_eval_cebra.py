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
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
def fit_circle_curvature(points, how=""):
    """
    Fit a circle to 3 consecutive points and return curvature (1/radius).
    If points are collinear or too close, return 0.
    """
    if(how == "triangle"):
    
        # Get three points
        points = points[::max(1,len(points)//3)]
        p1, p2, p3 = points[0], points[1], points[2]

        # Calculate the radius using the circumradius formula
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Area using Heron's formula
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        
        if area_squared <= 0:
            return 0  # Collinear points
        
        area = np.sqrt(area_squared)
         
        if area == 0:
            return 0
        
        # Radius = (a*b*c) / (4*Area)
        radius = (a * b * c) / (4 * area)
        if radius == 0:
            return 0
        return 1 / radius
    else:
        if(np.isnan(points).any()):
            return 0 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(points)
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        points_2d = pca.transform(X_scaled)       
        def circle_residuals(params, points):
            xc, yc, R = params
            distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
            return distances - R
        x = points_2d[:,0]
        y = points_2d[:,1]
        points = np.column_stack((x, y))
        x0 = [np.mean(x), np.mean(y), np.std(np.sqrt(x**2 + y**2))]
        try:
            res = least_squares(circle_residuals, x0, args=(points,))
        except ValueError as e:
            print(e)
            return 0
        if not res.success:
            return 0
        _, _, radius = res.x
        
        if(radius == 0):
            return 0 
        return 1/radius
 
def calculate_curvatures(trajectory):
    """Calculate curvature for each point in trajectory using sliding window."""
    offset = 20
    curvatures = []
    
    for i in range(len(trajectory)):
        points = trajectory[max(0,i-offset):min(i+offset, len(trajectory))]
        if len(points) >= 3:
            curvatures.append(fit_circle_curvature(points))
        else:
            curvatures.append(0)
    
    return np.array(curvatures)


PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
def get_phases(embryo_id, seq_len):
    annotation_file = os.path.join("embryo_dataset_annotations", f"{embryo_id}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    
    new_column += ["post_phase"] * (seq_len - len(new_column))
    new_column = new_column[:seq_len]
    
    return np.array([PHASES.index(phase) for phase in new_column]) 


def main(model_name):
    HOLDOUT = True
    GRADE = "TE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    login(os.getenv("HF_KEY"))
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/" + model_name)     
    model.to(device)
    VAL_EMBRYOS = pd.read_csv("embryo_dataset_grades.csv").rename(columns={"video_name":"embryo_id"}).dropna(subset=["ICM"])["embryo_id"].astype(str).tolist()
    grades_df = pd.read_csv(os.path.abspath("embryo_dataset_grades.csv")).rename(columns={"video_name":"embryo_id"}).dropna(subset=[GRADE])[["embryo_id",GRADE]]

    #latents_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    
    full_seq_df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})    
    full_seq_df = full_seq_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")
    full_seq_val_mask = full_seq_df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    full_seq_df_val = (full_seq_df[full_seq_val_mask] if HOLDOUT else full_seq_df) # just look at validation ICM embryos
    
    full_seq_df_train = (full_seq_df[~full_seq_val_mask] if HOLDOUT else full_seq_df) # just look at validation ICM embryos
    full_seq_dataset_train = IVFEmbryoDataset(full_seq_df_train, resize=128, norm="minmax01")

    
    full_seq_loader_train = DataLoader(
        full_seq_dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False 
    )
 
    cebra_time_model = CEBRA(model_architecture="offset10-model-mse",
                        batch_size=512,
                        learning_rate=2e-5,
                        temperature=10,
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
        grade_fig, grade_ax = plt.subplots(figsize=(20,20), subplot_kw={'projection': '3d'})
        all_x, all_y, all_z = [], [], [] 
        for grade,g_color in zip(["A", "B", "C"], [0,0.5,1]):
            grade_ds = IVFEmbryoDataset(full_seq_df[full_seq_df[GRADE] == grade], resize=128, norm="minmax01")
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
            d3_traj_ids = []
            for j, embryo_vol in enumerate(grade_loader):
                if(j >= len(axes)):
                    break
                d3_traj_ids.append(grade_ds.df.iloc[j]["embryo_id"])
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
            # track limits for the master plot
            all_x.extend([x for t in d3_trajs for x in t[:, 0]])
            all_y.extend([y for t in d3_trajs for y in t[:, 1]])
            all_z.extend([z for t in d3_trajs for z in t[:, 2]])
            im = None
            for i, d3_traj in enumerate(d3_trajs):
                im_grade = grade_ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=np.full((d3_traj.shape[0],), g_color), cmap='viridis', vmin=0, vmax=1)
                ax = axes[i]
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=np.linspace(0,1,d3_traj.shape[0]), cmap='viridis')
                
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim(z_lim) 
                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Time')
 
            fig.savefig(os.path.join("cebra_plots",f"{grade}.png"))
            plt.close(fig) 
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': '3d'})
            im = None
            for i, d3_traj in enumerate(d3_trajs):
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=np.linspace(0,1, d3_traj.shape[0]), cmap='viridis', vmin=0, vmax=1)
                
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
            plt.close(fig)
            # color by phases now
            fig, axes = plt.subplots(4, 4, figsize=(20, 20),subplot_kw={'projection': '3d'})
            axes = axes.ravel()
            im = None
            legend_elements = [Patch(facecolor=plt.cm.tab20c(i), label=phase) 
                   for i, phase in enumerate(PHASES)]

            for i, d3_traj in enumerate(d3_trajs):
                embryo_id = d3_traj_ids[i]
                c_phase = get_phases(embryo_id, len(d3_traj))
                ax = axes[i]
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=c_phase, cmap='tab20c', vmin=0, vmax=19)
                
                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            fig.legend(handles=legend_elements, title="Phases") 
            fig.savefig(os.path.join("cebra_plots",f"phases_{grade}.png"))
            plt.close(fig) 
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': '3d'})
            for i, d3_traj in enumerate(d3_trajs):
                embryo_id = d3_traj_ids[i]
                c_phase = get_phases(embryo_id, len(d3_traj))
                ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=c_phase, cmap='tab20c', vmin=0, vmax=19)
                
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim) 
            ax.set_xlabel("Cebra 1")
            ax.set_ylabel("Cebra 2")
            ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            fig.legend(handles=legend_elements, title="Phases") 
            fig.savefig(os.path.join("cebra_plots",f"phase_grouped_{grade}.png"))
            # now velocity
            plt.close(fig)
            fig, axes = plt.subplots(4, 4, figsize=(20, 20),subplot_kw={'projection': '3d'})
            axes = axes.ravel()
            im = None
            vels = []
            for i, d3_traj in enumerate(d3_trajs):
                vel = np.concatenate((np.diff(d3_traj, axis=0).mean(axis=1), np.array([0])))
                vel = vel / vel.max()
                vels.append(vel)
                ax = axes[i]
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=vel, cmap='viridis')
                
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim(z_lim) 
                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Velocity')
 
            fig.savefig(os.path.join("cebra_plots",f"vel_{grade}.png"))
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': '3d'})
            im = None
            for i, d3_traj in enumerate(d3_trajs):
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=vels[i], cmap='viridis')
                
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim) 
            ax.set_xlabel("Cebra 1")
            ax.set_ylabel("Cebra 2")
            ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Vel')
            fig.savefig(os.path.join("cebra_plots",f"vel_grouped_{grade}.png"))
            plt.close(fig)
            fig, axes = plt.subplots(4, 4, figsize=(20, 20),subplot_kw={'projection': '3d'})
            axes = axes.ravel()
            im = None
            curves = []
            for i, d3_traj in enumerate(d3_trajs):
                curv = calculate_curvatures(d3_traj)
                curv = curv / np.std(curv)
                curves.append(curv)
                ax = axes[i]
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=curv, cmap='viridis')
                
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim(z_lim) 
                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Curvature')
 
            fig.savefig(os.path.join("cebra_plots",f"curve_{grade}.png"))
            fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': '3d'})
            im = None
            for i, d3_traj in enumerate(d3_trajs):
                im = ax.scatter(d3_traj[:,0], d3_traj[:,1], d3_traj[:,2], c=curves[i], cmap='viridis')
                
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim) 
            ax.set_xlabel("Cebra 1")
            ax.set_ylabel("Cebra 2")
            ax.set_zlabel("Cebra 3")
                
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            fig.subplots_adjust(right=0.85) 
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label='Normalized Curvature')
            fig.savefig(os.path.join("cebra_plots",f"curve_grouped_{grade}.png"))
            plt.close(fig) 
        # do all the grade stuff now
        
        grade_ax.set_xlim(min(all_x), max(all_x))
        grade_ax.set_ylim(min(all_y), max(all_y))
        grade_ax.set_zlim(min(all_z), max(all_z))
    
        grade_ax.set_xlabel("Cebra 1")
        grade_ax.set_ylabel("Cebra 2")
        grade_ax.set_zlabel("Cebra 3")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        grade_fig.subplots_adjust(right=0.85) 
        cbar_ax = grade_fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap='viridis')
        cbar = grade_fig.colorbar(sm, cax=cbar_ax, label='Grade')
        cbar.set_ticks([0, 0.5, 0.99])
        cbar.set_ticklabels(['Grade A', 'Grade B', 'Grade C']) 

        grade_fig.savefig(os.path.join("cebra_plots","grouped_grade.png"))
        plt.close()



if __name__ == "__main__":
    import sys
    main(sys.argv[1])
