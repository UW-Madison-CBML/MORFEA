import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import seaborn as sns
from itertools import product
import umap
import matplotlib


import matplotlib.patches as patches
def main(model_name, grade):
    sig_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv"))
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"))
    
    sig_df = sig_df.rename(columns={"cell_id": "embryo_id", "video_name": "embryo_id"})
    grades_df = grades_df.rename(columns={"cell_id": "embryo_id", "video_name": "embryo_id"})
    
    if "embryo_id" not in sig_df.columns or "embryo_id" not in grades_df.columns:
        print("ERROR: Missing embryo_id column")
        print(sig_df.head())
        print(grades_df.head())
        raise ValueError("no embryo_id column")
    
    sig_cols = [col for col in sig_df.columns if col.startswith("s_")]
     
    df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")
    df = df[["embryo_id", grade] + sig_cols].dropna(subset=[grade])
    feature_cols = [col for col in df.columns if col.startswith('s_')]
    X = df[feature_cols].values
    grade_classes = ["A", "B", "C"]
    grades = np.array([grade_classes.index(val) for val in df[grade].values])
    X[np.isnan(X)] = 0
    print(f"Starting with {X.shape[0]} samples, {X.shape[1]} features\n")
    selector = VarianceThreshold(threshold=0) 
    X_filtered = selector.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    cmap = matplotlib.colors.ListedColormap(['green', 'yellow', 'red'])
    scatter = plt.scatter(embedding[:, 0], embedding[:,1], c=grades, cmap=cmap)

    handles, _ = scatter.legend_elements()

    plt.legend(handles, grade_classes, title="Grades", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Embryo Curvature Timestep Feature UMAP Embedding")
   
    plt.savefig(os.path.join("clusters", "umap.png"), dpi=300, bbox_inches='tight')
    plt.close() 
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python clustering_analysis.py <model_name> <grade_column>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    grade = sys.argv[2]
    main(model_name, grade)
