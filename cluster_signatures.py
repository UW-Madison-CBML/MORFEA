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

from matplotlib.patches import Patch
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
    grades = df[grade].values
     
    print(f"Starting with {X.shape[0]} samples, {X.shape[1]} features\n")
    selector = VarianceThreshold(threshold=0) 
    X_filtered = selector.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grade_colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    colors_grade = [grade_colors[g] for g in grades]
    axes[0].scatter(X_viz[:, 0], X_viz[:, 1], c=colors_grade, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Actual Grades (A=green, B=orange, C=red)')
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='A'),
                       Patch(facecolor='orange', edgecolor='black', label='B'),
                       Patch(facecolor='red', edgecolor='black', label='C')]
    axes[0].legend(handles=legend_elements)

    scatter = axes[1].scatter(X_viz[:, 0], X_viz[:, 1], c=clusters_combo, cmap='viridis', 
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('KMeans Clusters (Feature Selection + PCA)')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig('clusters/clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("=== STRATEGY 1: Feature Selection ===")
    selector = SelectKBest(f_classif, k=50)  
    print(grades)
    print(X_scaled)
    X_selected = selector.fit_transform(X_scaled, grades)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_selected = kmeans.fit_predict(X_selected)
    print(f"KMeans on selected features: Silhouette = {silhouette_score(X_selected, clusters_selected):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_selected)}")
    print(f"Grade vs Cluster agreement: {adjusted_rand_score(grades == 'C', clusters_selected):.3f}")

    print("\n=== STRATEGY 2: PCA Reduction ===")
    pca = PCA(n_components=0.80)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced to {X_pca.shape[1]} components (explaining {pca.explained_variance_ratio_.sum():.1%} variance)")

    kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    print(f"KMeans on PCA: Silhouette = {silhouette_score(X_pca, clusters_pca):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_pca)}")

    print("\n=== STRATEGY 3: Feature Selection + PCA ===")
    X_selected_pca = pca.fit_transform(X_selected)
    print(f"Selected + PCA: {X_selected_pca.shape[1]} components")

    kmeans_combo = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_combo = kmeans_combo.fit_predict(X_selected_pca)
    print(f"KMeans on selected+PCA: Silhouette = {silhouette_score(X_selected_pca, clusters_combo):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_combo)}")

    print("\n=== VISUALIZATION ===")

    pca_viz = PCA(n_components=2)
    X_viz = pca_viz.fit_transform(X_selected_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grade_colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    colors_grade = [grade_colors[g] for g in grades]
    axes[0].scatter(X_viz[:, 0], X_viz[:, 1], c=colors_grade, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Actual Grades (A=green, B=orange, C=red)')
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='A'),
                       Patch(facecolor='orange', edgecolor='black', label='B'),
                       Patch(facecolor='red', edgecolor='black', label='C')]
    axes[0].legend(handles=legend_elements)

    scatter = axes[1].scatter(X_viz[:, 0], X_viz[:, 1], c=clusters_combo, cmap='viridis', 
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('KMeans Clusters (Feature Selection + PCA)')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig('clusters/clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'clustering_comparison.png'")

    print("\n=== Cluster vs Grade Distribution ===")
    print(pd.crosstab(clusters_combo, grades, margins=True))    
    sig_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv"))
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=False)
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python clustering_analysis.py <model_name> <grade_column>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    grade = sys.argv[2]
    main(model_name, grade)
