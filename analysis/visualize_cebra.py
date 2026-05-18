import cebra
import torch
from cebra import CEBRA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg') 
from utils.geometric_features import calculate_curvatures, get_acc, get_vel
import numpy as np
import pandas as pd
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
GRADES = ["NA", "C", "B", "A"]
GRADE_COLORS = ["#888888", "#FF0000", "#FFFF00", "#00FF00"]

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
def plot_sequences(seqs, f_name, c=None, cmap='viridis', uniform_bounds=False, cbar_label="Time", log_scale=False):
    if(c is None):
        c = [np.linspace(0,1,len(seq)) for seq in seqs]
    x_list = [x for seq in seqs for x in seq[:, 0]]
    y_list = [y for seq in seqs for y in seq[:, 1]]
    z_list = [z for seq in seqs for z in seq[:, 2]]
    x_lim = (min(x_list),max(x_list))
    y_lim = (min(y_list),max(y_list))
    z_lim = (min(z_list),max(z_list))

    least_greater_square = max(2, int(math.sqrt(len(seqs)) + 1))
    grid_fig, grid_axes = plt.subplots(least_greater_square, least_greater_square,figsize=(5*least_greater_square, 5*least_greater_square), subplot_kw={'projection': '3d'})
    grid_axes = grid_axes.ravel()
    grid_im = None
    group_fig, group_ax = plt.subplots(subplot_kw={'projection': '3d'})
    group_im = None
    for i, seq in enumerate(seqs):
        individ_fig, individ_ax = plt.subplots(subplot_kw={'projection':'3d'})
        individ_im = None
        if(cmap =="phase"):
            grid_im = grid_axes[i].scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap='tab20c', vmin=0, vmax=19)
            individ_im = individ_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap='tab20c', vmin=0, vmax=19)
            group_im = group_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap='tab20c', vmin=0, vmax=19)
        elif(log_scale): # discrete colorbar and log colorbar are mutually exclusive
            grid_im = grid_axes[i].scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap, norm=matplotlib.colors.LogNorm())
            individ_im = individ_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap, norm=matplotlib.colors.LogNorm())
            group_im = group_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap, norm=matplotlib.colors.LogNorm())
        else:
            grid_im = grid_axes[i].scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap)
            individ_im = individ_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap)
            group_im = group_ax.scatter(seq[:,0], seq[:,1], seq[:,2], c=c[i], cmap=cmap)

        individ_ax.set_xlabel("Cebra 1")
        individ_ax.set_ylabel("Cebra 2")
        individ_ax.set_zlabel("Cebra 3")
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        individ_fig.subplots_adjust(right=0.85) 
        if(cmap == "phase"):
            legend_elements = [Patch(facecolor=plt.cm.tab20c(i), label=phase) for i, phase in enumerate(PHASES)]
            individ_fig.legend(handles=legend_elements, title="Phases") 
        else:
            cbar_ax = individ_fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
            if individ_im is not None:
                individ_fig.colorbar(individ_im, cax=cbar_ax, label=cbar_label)

        individ_fig.savefig(os.path.join("cebra_plots",f"individ-{f_name}-{i}.png"))
        plt.close(individ_fig) 
        if(uniform_bounds):
            grid_axes[i].set_xlim(x_lim)
            grid_axes[i].set_ylim(y_lim)
            grid_axes[i].set_zlim(z_lim) 
        grid_axes[i].set_xlabel("Cebra 1")
        grid_axes[i].set_ylabel("Cebra 2")
        grid_axes[i].set_zlabel("Cebra 3")
    
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    grid_fig.subplots_adjust(right=0.85) 
    if(cmap == "phase"):
        legend_elements = [Patch(facecolor=plt.cm.tab20c(i), label=phase) for i, phase in enumerate(PHASES)]
        grid_fig.legend(handles=legend_elements, title="Phases") 
    else:
        cbar_ax = grid_fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
        if grid_im is not None:
            grid_fig.colorbar(grid_im, cax=cbar_ax, label=cbar_label)

    grid_fig.savefig(os.path.join("cebra_plots",f"grid-{f_name}.png"))
    plt.close(grid_fig)   
    
    group_ax.set_xlim(x_lim)
    group_ax.set_ylim(y_lim)
    group_ax.set_zlim(z_lim) 
    group_ax.set_xlabel("Cebra 1")
    group_ax.set_ylabel("Cebra 2")
    group_ax.set_zlabel("Cebra 3")
    group_fig.subplots_adjust(right=0.85) 
    if(cmap == "phase"):
        legend_elements = [Patch(facecolor=plt.cm.tab20c(i), label=phase) for i, phase in enumerate(PHASES)]
        group_fig.legend(handles=legend_elements, title="Phases") 
    else:
        cbar_ax = group_fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
        if group_im is not None:
            group_fig.colorbar(group_im, cax=cbar_ax, label=cbar_label)

    group_fig.savefig(os.path.join("cebra_plots",f"group-{f_name}.png"))
    plt.close(group_fig)


def main(model_name, image_name, grade_args, phase_args):
    
    GRADE = "TE"

    # latents df is also metadata for the cebra embeddings
    latents_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    cebra_np = np.load(os.path.join("cebra_latents", f"{model_name}.npy"))
    cebra_df = pd.DataFrame(cebra_np, columns=["z_0","z_1","z_2"])
    df = pd.concat([latents_df, cebra_df], axis=1)
    if("NA" not in grade_args):
        df = df.dropna(subset=[GRADE])
    else:
        df = df.fillna("NA")

    df = df[df['phase'].str.contains("|".join(phase_args),regex=True)]
    print(model_name) 
    max_imgs = 200
    # do one colored by grades
    seqs = []
    c = []
    embryo_grade_groups = []
    for g in grade_args:
        embryo_groups = [group for _, group in df[df[GRADE] == g].groupby("embryo_id") ]
        np.random.shuffle(embryo_groups)
        embryo_groups = embryo_groups[:max_imgs]
        embryo_grade_groups.append(embryo_groups)
        for group in embryo_groups:
            seqs.append(group[["z_0","z_1","z_2"]].to_numpy())
            c.append([GRADE_COLORS[GRADES.index(g)]] * len(group))
            
    plot_sequences(seqs, f"grade_{image_name}", c=c, cmap=None, cbar_label="Grade")
    # do a bunch per grade using diff colormaps
    for g, embryo_groups in zip(grade_args, embryo_grade_groups):
        #--------------------------------
        # time
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[["z_0","z_1","z_2"]].to_numpy())
            c.append(group['time_step'].to_numpy()/group['time_step'].max()) # since we are removing some phases we need to use ground_truth time not inherent order of df
            
            
        plot_sequences(seqs, f"time_{g}_{image_name}",c=c, cbar_label="Time")
        #--------------------------------
        # phase
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[["z_0","z_1","z_2"]].to_numpy())
            c.append([PHASES.index(p) for p in group['phase']])
            
        plot_sequences(seqs, f"phase_{g}_{image_name}", c=c, cmap="phase", cbar_label="Phase")
        #--------------------------------
        # curvature
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[["z_0","z_1","z_2"]].to_numpy()
            seqs.append(seq)
            c.append(calculate_curvatures(seq, offset=13))
            
        plot_sequences(seqs, f"curv_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True)

        
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Plot cebra latents',
                    description='',
                    epilog='')
    
    parser.add_argument('model_name')           
    parser.add_argument('image_name')           
    parser.add_argument('--all-grades', action='store_true')
    parser.add_argument('--all-phases', action='store_true')
    parser.add_argument("--phases", action="extend", nargs="+", type=str) 
    parser.add_argument("--grades", action="extend", nargs="+", type=str) 
    args = parser.parse_args()
    main(args.model_name, args.image_name, ["A","B","C"] if args.all_grades else args.grades, PHASES if args.all_phases else args.phases)
