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
def main():
    sig_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv")).rename(columns={"embryo_id":"cell_id"})
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=False)
    sig_df = sig_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    grades_df = grades_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    if(not("embryo_id" in sig_df.columns and "embryo_id" in grades_df.columns)):
        print(sig_df.head())
        print(grades_df.head())
        raise ValueError("no embryo_id column")
    sig_cols = [col for col in sig_df.columns if col.startswith("s_")]
    df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", "TE"] + sig_cols].dropna(subset=["TE"])

    # Extract feature columns (assuming they start with 's_')
    feature_cols = [col for col in df.columns if col.startswith('s_')]
    X = df[feature_cols].values

    # Standardize the features (important for KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform KMeans clustering into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster assignments to the dataframe
    df['Cluster'] = clusters

    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a comprehensive visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Clusters
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('KMeans Clusters (Unsupervised)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot 2: Actual grades
    grade_colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    grades = df['TE'].values
    colors = [grade_colors[g] for g in grades]
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100, 
                               alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].set_title('Actual Grades')

    # Add legend for grades
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='A'),
                       Patch(facecolor='orange', edgecolor='black', label='B'),
                       Patch(facecolor='red', edgecolor='black', label='C')]
    axes[1].legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig('clusters/clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'clustering_comparison.png'")

    # Optional: Print a confusion-style matrix showing cluster vs grade distribution
    print("\nCluster vs Grade Distribution:")
    print(pd.crosstab(df['Cluster'], df['TE'], margins=True))

    # Calculate silhouette score to assess clustering quality
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"\nSilhouette Score: {silhouette_avg:.3f}")
    print("(Score ranges from -1 to 1; higher is better, >0.5 is generally good)")

    # Optional: Adjusted Rand Index if you want to measure agreement between clusters and grades
    from sklearn.metrics import adjusted_rand_score
    grade_numeric = pd.factorize(df['TE'])[0]  # Convert grades to numbers
    ari = adjusted_rand_score(grade_numeric, clusters)
    print(f"Adjusted Rand Index (cluster vs grade): {ari:.3f}")
    print("(Ranges from -1 to 1; higher means better agreement)")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)
