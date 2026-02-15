import os
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
def main(model_name):
    sig_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv")).rename(columns={"embryo_id":"cell_id"})
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"))
    sig_df = sig_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    grades_df = grades_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    if(not("embryo_id" in sig_df.columns and "embryo_id" in grades_df.columns)):
        print(sig_df.head())
        print(grades_df.head())
        raise ValueError("no embryo_id column")
    sig_cols = [col for col in sig_df.columns if col.startswith("s_")]
    df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", grade] + sig_cols].dropna(subset=[grade])

    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('s_')]
    X = df[feature_cols].values
    grades = df[grade].values

    print(f"Starting with {X.shape[0]} samples, {X.shape[1]} features\n")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Strategy 1: Use only the most informative features
    print("=== STRATEGY 1: Feature Selection ===")
    selector = SelectKBest(f_classif, k=50)  # Keep top 50 features
    X_selected = selector.fit_transform(X_scaled, grades)
    print(f"Selected top 50 features (was 500)")

    # Now cluster on selected features
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_selected = kmeans.fit_predict(X_selected)
    print(f"KMeans on selected features: Silhouette = {silhouette_score(X_selected, clusters_selected):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_selected)}")
    print(f"Grade vs Cluster agreement: {adjusted_rand_score(grades == 'C', clusters_selected):.3f}")

    # Strategy 2: PCA dimensionality reduction
    print("\n=== STRATEGY 2: PCA Reduction ===")
    # Keep enough components to explain 80% variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced to {X_pca.shape[1]} components (explaining {pca.explained_variance_ratio_.sum():.1%} variance)")

    kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    print(f"KMeans on PCA: Silhouette = {silhouette_score(X_pca, clusters_pca):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_pca)}")
    # Strategy 3: Combine both - select features, then reduce
    print("\n=== STRATEGY 3: Feature Selection + PCA ===")
    X_selected_pca = pca.fit_transform(X_selected)
    print(f"Selected + PCA: {X_selected_pca.shape[1]} components")

    kmeans_combo = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_combo = kmeans_combo.fit_predict(X_selected_pca)
    print(f"KMeans on selected+PCA: Silhouette = {silhouette_score(X_selected_pca, clusters_combo):.3f}")
    print(f"Cluster distribution: {np.bincount(clusters_combo)}")
    # Visualize the best approach
    print("\n=== VISUALIZATION ===")

    # Use Strategy 3 for visualization (best of both worlds)
    pca_viz = PCA(n_components=2)
    print(X_selected_pca.shape)
    X_viz = pca_viz.fit_transform(X_selected_pca) if X_selected_pca.shape[1] >= 2 else X_selected_pca

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual grades
    grade_colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    colors_grade = [grade_colors[g] for g in grades]
    axes[0].scatter(X_viz[:, 0], X_viz[:, 1], c=colors_grade, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Actual Grades (A=green, B=orange, C=red)')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='A'),
                       Patch(facecolor='orange', edgecolor='black', label='B'),
                       Patch(facecolor='red', edgecolor='black', label='C')]
    axes[0].legend(handles=legend_elements)

    # Clusters
    scatter = axes[1].scatter(X_viz[:, 0], X_viz[:, 1], c=clusters_combo, cmap='viridis', 
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('KMeans Clusters (Feature Selection + PCA)')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig('clusters/clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'clustering_comparison.png'")

    # Show cluster vs grade breakdown
    print("\n=== Cluster vs Grade Distribution ===")
    print(pd.crosstab(clusters_combo, grades, margins=True))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name, "TE")
    main(args.name, "ICM")
