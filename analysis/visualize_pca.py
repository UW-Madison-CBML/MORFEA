import cebra
import torch
from cebra import CEBRA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg') 
from geometric_features import calculate_curvatures, get_acc, get_vel
import numpy as np
import pandas as pd
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import math
from visualize_cebra import plot_sequences

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

def main(model_name, image_name, grade_args, phase_args):
    rng = np.random.default_rng(seed=42)
    GRADE = "TE"
    PCA_COLS = ["pca_0","pca_1","pca_2"]
    UMAP_COLS = ["umap_0", "umap_1", "umap_2"]
    # latents df is also metadata for the cebra embeddings
    latents_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    latents_np = np.load(os.path.join("latents", f"{model_name}.npy"))
    pca = PCA(n_components=3)
    standard_scaler = StandardScaler()
    pca_lats = pca.fit_transform(standard_scaler.fit_transform(latents_np))
    pca_df = pd.DataFrame(pca_lats, columns = PCA_COLS, index=latents_df.index)
    umap = UMAP(n_components=3)
    standard_scaler = StandardScaler()
    umap_lats = umap.fit_transform(latents_np)
    umap_df = pd.DataFrame(umap_lats, columns = UMAP_COLS, index=latents_df.index)
    df = pd.concat([latents_df,pca_df, umap_df], axis=1)
    if("NA" not in grade_args):
        df = df.dropna(subset=[GRADE])
    else:
        df = df.fillna("NA")

    df = df[df['phase'].str.contains("|".join(phase_args),regex=True)]
    print(model_name) 
    max_imgs = -1
    # do one colored by grades
    seqs = []
    c = []
    embryo_grade_groups = []
    embryo_grade_groups_names = []
    
    names = []
    for g in grade_args:
        embryo_groups_names = list(df[df[GRADE] == g].groupby("embryo_id"))
        rng.shuffle(embryo_groups)
        if max_imgs != -1:
            embryo_groups_names = embryo_groups_names[:max_imgs]
        embryo_names, embryo_groups = zip(*embryo_groups_names)
        embryo_grade_groups.append(embryo_groups)
        embryo_grade_groups_names.append(embryo_names)
        
        for name, group in zip(embryo_names, embryo_groups):
            seqs.append(group[0][PCA_COLS].to_numpy())
            c.append([GRADE_COLORS[GRADES.index(g)]] * len(group))
            names.append(name)
            
    plot_sequences(seqs, f"grade_{image_name}", c=c, cmap=None, cbar_label="Grade", folder="pca_plots", axlabel="PCA", individ_names=names, axes_off=True)
    # do a bunch per grade using diff colormaps
    for g, embryo_groups, names in zip(grade_args, embryo_grade_groups, embryo_grade_groups_names):
        #--------------------------------
        # time
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[PCA_COLS].to_numpy())
            c.append(group['time_step'].to_numpy()/group['time_step'].max()) # since we are removing some phases we need to use ground_truth time not inherent order of df
            
            
        plot_sequences(seqs, f"time_{g}_{image_name}",c=c, cbar_label="Time", folder="pca_plots", axlabel="PCA", individ_names=names, axes_off=True)
        #--------------------------------
        # phase
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[PCA_COLS].to_numpy())
            c.append([PHASES.index(p) for p in group['phase']])
            
        plot_sequences(seqs, f"phase_{g}_{image_name}", c=c, cmap="phase", cbar_label="Phase", folder="pca_plots", axlabel="PCA",axes_off=True, individ_names= names)
        #--------------------------------
        # curvature
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PCA_COLS].to_numpy()
            seqs.append(seq)
            c.append(calculate_curvatures(seq, offset=13))
            
        plot_sequences(seqs, f"curv_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="PCA",axes_off=True, individ_names=names)
        #--------------------------------
        # acceleration
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PCA_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_acc(seq))
            
        plot_sequences(seqs, f"acc_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="PCA",axes_off=True, individ_names = names)

        #--------------------------------
        # velocity
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PCA_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_vel(seq))
            
        plot_sequences(seqs, f"umap_vel_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="PCA", axes_off=True, individ_names = names)
    # --------------------------------------------
    # --------------------------------------------
    # UMAP
    seqs = []
    c = []
    names = []
    for embryo_groups,groups_names, g in zip(embryo_grade_groups, embryo_grade_groups_names, grade_args):
        names += groups_names
        for group in embryo_groups:
            seqs.append(group[UMAP_COLS].to_numpy())
            c.append([GRADE_COLORS[GRADES.index(g)]] * len(group))

            
    plot_sequences(seqs, f"umap_grade_{image_name}", c=c, cmap=None, cbar_label="Grade", folder="pca_plots", axlabel="UMAP",axes_off=True, individ_names = names)
    # do a bunch per grade using diff colormaps
    for g, embryo_groups, names in zip(grade_args, embryo_grade_groups, embryo_grade_groups_names):
        #--------------------------------
        # time
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[UMAP_COLS].to_numpy())
            c.append(group['time_step'].to_numpy()/group['time_step'].max()) # since we are removing some phases we need to use ground_truth time not inherent order of df
            
            
        plot_sequences(seqs, f"umap_time_{g}_{image_name}",c=c, cbar_label="Time", folder="pca_plots", axlabel="UMAP",axes_off=True, individ_names = names)
        #--------------------------------
        # phase
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[UMAP_COLS].to_numpy())
            c.append([PHASES.index(p) for p in group['phase']])
            
        plot_sequences(seqs, f"umap_phase_{g}_{image_name}", c=c, cmap="phase", cbar_label="Phase", folder="pca_plots", axlabel="UMAP", axes_off=True, individ_names = names)
        #--------------------------------
        # curvature
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[UMAP_COLS].to_numpy()
            seqs.append(seq)
            c.append(calculate_curvatures(seq, offset=13))
            
        plot_sequences(seqs, f"umap_curv_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="UMAP", axes_off=True, individ_names = names)
        #--------------------------------
        # acceleration
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[UMAP_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_acc(seq))
            
        plot_sequences(seqs, f"umap_acc_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="UMAP", axes_off=True, individ_names = names)

        #--------------------------------
        # velocity
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[UMAP_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_vel(seq))
            
        plot_sequences(seqs, f"umap_vel_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="UMAP", axes_off=True, individ_names = names)




        
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Plot pca latents',
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
