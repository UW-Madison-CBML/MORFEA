import cebra
import os
import torch
from cebra import CEBRA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../utils")) # for non-CHTC use
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../training")) # for non-CHTC use
matplotlib.use('Agg') 
from geometric_features import calculate_curvatures, get_acc, get_vel
import numpy as np
import pandas as pd
from huggingface_hub import login
from torch.utils.data import DataLoader
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import math
from visualize_cebra import plot_sequences
from tqdm import tqdm
import phate
from stage_dataset import StageDataset

GRADES = ["NA", "C", "B", "A"]
GRADE_COLORS = ["#888888", "#FF0000", "#FFFF00", "#00FF00"]

def main(model_name, image_name, grade_args, phase_args, two_d=True):
    rng = np.random.default_rng(seed=42)
    GRADE = "TE"
    PHATE_COLS = ["pca_0","pca_1"]
    # latents df is also metadata for the cebra embeddings
    latents_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    latents_np = np.load(os.path.join("latents", f"{model_name}.npy"))
    mask = latents_df['phase'].isin(phase_args) & latents_df['TE'].isin(grade_args)
    latents_df = latents_df[mask]
    latents_np = latents_np[mask]
    phate_model = phate.PHATE(knn=5, decay=15, n_jobs=-2, verbose=True)
    phate_lats = phate_model.fit_transform(latents_np)
    phate_df = pd.DataFrame(phate_lats, columns = PHATE_COLS, index=latents_df.index)
    df = pd.concat([latents_df,phate_df], axis=1)
    if("NA" not in grade_args):
        df = df.dropna(subset=[GRADE])
    else:
        df = df.fillna("NA")

    max_imgs = 20
    # do one colored by grades
    seqs = []
    c = []
    embryo_grade_groups = []
    embryo_grade_groups_names = []
    
    names = []
    for g in grade_args:
        embryo_groups_names = list(df[df[GRADE] == g].groupby("embryo_id"))
        rng.shuffle(embryo_groups_names)
        if max_imgs != -1:
            embryo_groups_names = embryo_groups_names[:max_imgs]
        embryo_names, embryo_groups = zip(*embryo_groups_names)
        embryo_grade_groups.append(embryo_groups)
        embryo_grade_groups_names.append(embryo_names)
        
        for name, group in zip(embryo_names, embryo_groups):
            seqs.append(group[PHATE_COLS].to_numpy())
            c.append([GRADE_COLORS[GRADES.index(g)]] * len(group))
            names.append(name)
            
    plot_sequences(seqs, f"grade_{image_name}", c=c, cmap="grade", cbar_label="Grade", folder="pca_plots", axlabel="PCA", individ_names=names, axis_off=True, two_d=two_d)
    # do a bunch per grade using diff colormaps
    for g, embryo_groups, names in tqdm(zip(grade_args, embryo_grade_groups, embryo_grade_groups_names)):
        #--------------------------------
        # time
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[PHATE_COLS].to_numpy())
            c.append(group['time_step'].to_numpy()/group['time_step'].max()) # since we are removing some phases we need to use ground_truth time not inherent order of df
            
            
        plot_sequences(seqs, f"time_{g}_{image_name}",c=c, cbar_label="Time", folder="pca_plots", axlabel="PCA", individ_names=names, axis_off=True, two_d=two_d)
        #--------------------------------
        # embryo_id
        seqs = []
        c = []
        for i, group in enumerate(embryo_groups):
            seqs.append(group[PHATE_COLS].to_numpy())
            c.append([i / len(embryo_groups)] * len(group))
            
        plot_sequences(seqs, f"embryo_id_{g}_{image_name}", c=c, folder="pca_plots", axlabel="PCA", individ_names=names, axis_off=True, vminmax=[0,1], two_d=two_d)

        #--------------------------------
        # phase
        seqs = []
        c = []
        for group in embryo_groups:
            seqs.append(group[PHATE_COLS].to_numpy())
            c.append([StageDataset.PHASES.index(p) for p in group['phase']])
            
        plot_sequences(seqs, f"phase_{g}_{image_name}", c=c, cmap="phase", cbar_label="Phase", folder="pca_plots", axlabel="PCA", axis_off=True, individ_names= names, two_d=two_d)
        #--------------------------------
        # curvature
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PHATE_COLS].to_numpy()
            seqs.append(seq)
            c.append(calculate_curvatures(seq, offset=13))
            
        plot_sequences(seqs, f"curv_{g}_{image_name}", c=c, cbar_label="Curv", log_scale=True, folder="pca_plots", axlabel="PCA",axis_off=True, individ_names=names, two_d=two_d)
        #--------------------------------
        # acceleration
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PHATE_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_acc(seq))
            
        plot_sequences(seqs, f"acc_{g}_{image_name}", c=c, cbar_label="Acc", log_scale=True, folder="pca_plots", axlabel="PCA",axis_off=True, individ_names = names, two_d=two_d)

        #--------------------------------
        # velocity
        seqs = []
        c = []
        for group in embryo_groups:
            seq = group[PHATE_COLS].to_numpy()
            seqs.append(seq)
            c.append(get_vel(seq))
            
        plot_sequences(seqs, f"vel_{g}_{image_name}", c=c, cbar_label="Vel", log_scale=True, folder="pca_plots", axlabel="PCA", axis_off=True, individ_names = names, two_d=two_d)
    # --------------------------------------------
    # --------------------------------------------
            
    
    
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
