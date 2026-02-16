import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import seaborn as sns
from itertools import product

def calculate_cluster_purity(clusters, grades):
    """
    Calculate cluster purity: for each cluster, what % is the dominant grade?
    
    Returns:
        purity: weighted average purity across all clusters
        cluster_purities: dict mapping cluster_id -> purity score
        dominant_grades: dict mapping cluster_id -> dominant grade
    """
    cluster_purities = {}
    dominant_grades = {}
    total_samples = len(clusters)
    weighted_purity = 0.0
    
    for cluster_id in np.unique(clusters):
        # Get all grades in this cluster
        cluster_mask = clusters == cluster_id
        cluster_grades = grades[cluster_mask]
        cluster_size = len(cluster_grades)
        
        # Find most common grade and its frequency
        unique, counts = np.unique(cluster_grades, return_counts=True)
        max_idx = np.argmax(counts)
        dominant_grade = unique[max_idx]
        dominant_count = counts[max_idx]
        
        # Purity is the fraction of dominant grade in this cluster
        purity = dominant_count / cluster_size
        
        cluster_purities[cluster_id] = purity
        dominant_grades[cluster_id] = dominant_grade
        
        # Weight by cluster size
        weighted_purity += purity * (cluster_size / total_samples)
    
    return weighted_purity, cluster_purities, dominant_grades


def main(model_name, grade):
    # Load data
    sig_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv"))
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"))
    
    # Standardize column names - do this once properly
    sig_df = sig_df.rename(columns={"cell_id": "embryo_id", "video_name": "embryo_id"})
    grades_df = grades_df.rename(columns={"cell_id": "embryo_id", "video_name": "embryo_id"})
    
    if "embryo_id" not in sig_df.columns or "embryo_id" not in grades_df.columns:
        print("ERROR: Missing embryo_id column")
        print(sig_df.head())
        print(grades_df.head())
        raise ValueError("no embryo_id column")
    
    # Get signature columns
    sig_cols = [col for col in sig_df.columns if col.startswith("s_")]
    
    # Merge - keep ALL samples (graded and ungraded)
    df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")
    df = df[["embryo_id", grade] + sig_cols]
    
    # Extract features (ALL samples)
    X = df[sig_cols].values
    
    # Track which samples are graded (mask for evaluation)
    graded_mask = df[grade].notna().values
    grades_all = df[grade].values  # Contains NaN for ungraded
    grades_graded_only = df[grade][graded_mask].values  # Only graded samples
    
    print(f"Total samples: {X.shape[0]} ({graded_mask.sum()} graded, {(~graded_mask).sum()} ungraded)")
    print(f"Features: {X.shape[1]}")
    print(f"Graded distribution: {dict(zip(*np.unique(grades_graded_only, return_counts=True)))}\n")
    
    # Standardize once at the beginning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create output directory
    os.makedirs('clusters', exist_ok=True)
    
    # Define parameter grid
    feature_selection_methods = [
        ('none', None),
        ('kbest_10', SelectKBest(f_classif, k=10)),
        ('kbest_25', SelectKBest(f_classif, k=25)),
        ('kbest_50', SelectKBest(f_classif, k=50)),
        ('kbest_100', SelectKBest(f_classif, k=min(100, X.shape[1]))),
        ('kbest_200', SelectKBest(f_classif, k=min(200, X.shape[1]))),
        ('mutual_info_25', SelectKBest(mutual_info_classif, k=25)),
        ('mutual_info_50', SelectKBest(mutual_info_classif, k=50)),
        ('mutual_info_100', SelectKBest(mutual_info_classif, k=min(100, X.shape[1]))),
    ]
    
    pca_configs = [
        ('none', None),
        ('pca_80', PCA(n_components=0.80)),
        ('pca_90', PCA(n_components=0.90)),
        ('pca_95', PCA(n_components=0.95)),
        ('pca_50d', PCA(n_components=min(50, X.shape[1]-1))),
        ('pca_20d', PCA(n_components=min(20, X.shape[1]-1))),
        ('pca_10d', PCA(n_components=min(10, X.shape[1]-1))),
    ]
    
    cluster_counts = [2, 3, 4, 5, 6, 8]
    
    # Store all results
    results = []
    
    print(f"Testing {len(feature_selection_methods)} × {len(pca_configs)} × {len(cluster_counts)} = {len(feature_selection_methods) * len(pca_configs) * len(cluster_counts)} configurations\n")
    print("=" * 80)
    
    config_num = 0
    total_configs = len(feature_selection_methods) * len(pca_configs) * len(cluster_counts)
    
    for (feat_name, feat_selector), (pca_name, pca_model), n_clusters in product(
        feature_selection_methods, pca_configs, cluster_counts
    ):
        config_num += 1
        config_label = f"{feat_name}__{pca_name}__{n_clusters}clusters"
        
        print(f"[{config_num}/{total_configs}] {config_label}")
        
        try:
            # Apply feature selection (use ALL data)
            if feat_selector is not None:
                # For supervised selectors, only fit on graded samples
                X_graded_for_selection = X_scaled[graded_mask]
                feat_selector.fit(X_graded_for_selection, grades_graded_only)
                # But transform ALL samples
                X_transformed = feat_selector.transform(X_scaled)
                n_features = X_transformed.shape[1]
            else:
                X_transformed = X_scaled.copy()
                n_features = X_transformed.shape[1]
            
            # Apply PCA (on ALL data)
            if pca_model is not None:
                X_transformed = pca_model.fit_transform(X_transformed)
                n_components = X_transformed.shape[1]
                variance_explained = pca_model.explained_variance_ratio_.sum()
            else:
                n_components = X_transformed.shape[1]
                variance_explained = 1.0
            
            # Cluster ALL data (graded + ungraded)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            clusters_all = kmeans.fit_predict(X_transformed)
            
            # Extract clusters for graded samples only
            clusters_graded = clusters_all[graded_mask]
            X_transformed_graded = X_transformed[graded_mask]
            
            # Calculate metrics ONLY on graded samples
            silhouette = silhouette_score(X_transformed_graded, clusters_graded)
            davies_bouldin = davies_bouldin_score(X_transformed_graded, clusters_graded)
            
            # Calculate ARI for each grade vs clusters
            grade_unique = np.unique(grades_graded_only)
            ari_scores = {}
            for g in grade_unique:
                ari = adjusted_rand_score(grades_graded_only == g, clusters_graded)
                ari_scores[f'ari_{g}'] = ari
            
            # Overall clustering quality with grades
            ari_overall = adjusted_rand_score(grades_graded_only, clusters_graded)
            
            # Calculate cluster purity (how homogeneous each cluster is)
            purity, cluster_purities, dominant_grades = calculate_cluster_purity(clusters_graded, grades_graded_only)
            
            # Store results
            result = {
                'config': config_label,
                'feature_method': feat_name,
                'pca_method': pca_name,
                'n_clusters': n_clusters,
                'n_features': n_features,
                'n_components': n_components,
                'variance_explained': variance_explained,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'ari_overall': ari_overall,
                'purity': purity,
                'dominant_grades': dominant_grades,
                **ari_scores
            }
            results.append(result)
            
            print(f"  Features: {n_features} → Components: {n_components}")
            print(f"  Silhouette: {silhouette:.3f} | Davies-Bouldin: {davies_bouldin:.3f} | ARI: {ari_overall:.3f}")
            print(f"  Purity (graded only): {purity:.3f} | Dominant grades: {dominant_grades}")
            print(f"  All cluster sizes: {dict(zip(*np.unique(clusters_all, return_counts=True)))}")
            print(f"  Graded cluster sizes: {dict(zip(*np.unique(clusters_graded, return_counts=True)))}")
            
            # Create visualization
            create_cluster_visualization(
                X_transformed, clusters_all, X_transformed_graded, clusters_graded, 
                grades_graded_only, config_label, 
                silhouette, davies_bouldin, ari_overall, purity, n_clusters
            )
            
            # Create confusion matrix (only for graded samples)
            create_confusion_matrix(clusters_graded, grades_graded_only, config_label, n_clusters)
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('purity', ascending=False)
    results_df.to_csv('clusters/all_results.csv', index=False)
    
    # Also save versions sorted by other metrics
    results_df.sort_values('silhouette', ascending=False).to_csv('clusters/results_by_silhouette.csv', index=False)
    results_df.sort_values('ari_overall', ascending=False).to_csv('clusters/results_by_ari.csv', index=False)
    results_df.sort_values('davies_bouldin', ascending=True).to_csv('clusters/results_by_davies_bouldin.csv', index=False)
    
    # Create summary report
    create_summary_report(results_df, len(grades_graded_only))
    
    print("=" * 80)
    print(f"\nCompleted! Results saved to clusters/")
    print(f"Total configurations tested: {len(results_df)}")
    print(f"\nTop 10 by Cluster Purity (grade agreement):")
    print(results_df[['config', 'purity', 'silhouette', 'ari_overall', 'n_clusters', 'dominant_grades']].head(10).to_string(index=False))
    print(f"\nTop 10 by Silhouette Score:")
    results_by_sil = results_df.sort_values('silhouette', ascending=False)
    print(results_by_sil[['config', 'purity', 'silhouette', 'ari_overall', 'n_clusters']].head(10).to_string(index=False))


def create_cluster_visualization(X_transformed_all, clusters_all, X_transformed_graded, 
                                  clusters_graded, grades_graded, config_label, 
                                  silhouette, davies_bouldin, ari, purity, n_clusters):
    """Create side-by-side visualization of actual grades and clusters"""
    
    # If more than 2D, use PCA to visualize ALL data
    if X_transformed_all.shape[1] > 2:
        pca_viz = PCA(n_components=2)
        X_viz_all = pca_viz.fit_transform(X_transformed_all)
        var_explained = pca_viz.explained_variance_ratio_
    else:
        X_viz_all = X_transformed_all
        var_explained = [1.0, 0.0] if X_transformed_all.shape[1] == 1 else [1.0, 1.0]
    
    # Also transform graded subset using same PCA
    if X_transformed_all.shape[1] > 2:
        X_viz_graded = pca_viz.transform(X_transformed_graded)
    else:
        X_viz_graded = X_transformed_graded
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Actual grades (only graded samples, with ungraded in background)
    grade_colors_map = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
    
    # Plot all samples in gray (ungraded background)
    axes[0].scatter(X_viz_all[:, 0], X_viz_all[:, 1], c='lightgray', s=50, alpha=0.3, 
                    edgecolors='none', label='Ungraded')
    
    # Overlay graded samples with colors
    colors_grade = [grade_colors_map.get(g, '#95a5a6') for g in grades_graded]
    axes[0].scatter(X_viz_graded[:, 0], X_viz_graded[:, 1], c=colors_grade, s=100, alpha=0.8, 
                    edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    axes[0].set_title(f'Actual Grades ({len(grades_graded)} graded)', fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=label) 
                       for label, color in grade_colors_map.items()]
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='none', label='Ungraded'))
    axes[0].legend(handles=legend_elements, loc='best')
    
    # Right plot: Clusters (ALL samples)
    scatter = axes[1].scatter(X_viz_all[:, 0], X_viz_all[:, 1], c=clusters_all, cmap='viridis', 
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    axes[1].set_title(f'KMeans Clusters (k={n_clusters}, all {len(clusters_all)} samples)', 
                      fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')
    
    # Add metrics as text (based on graded subset only)
    metrics_text = f'Metrics (graded only):\nPurity: {purity:.3f}\nSilhouette: {silhouette:.3f}\nDavies-Bouldin: {davies_bouldin:.3f}\nARI: {ari:.3f}'
    axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(config_label, fontsize=12, y=1.00)
    plt.tight_layout()
    
    filename = f'clusters/viz_{config_label}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def create_confusion_matrix(clusters, grades, config_label, n_clusters):
    """Create confusion matrix heatmap of cluster vs grade"""
    
    # Create crosstab
    ct = pd.crosstab(clusters, grades, margins=False)
    
    # Normalize by row (show percentage within each cluster)
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # Calculate purity for each cluster (max percentage in each row)
    cluster_purities = ct_norm.max(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], 
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Grade', fontsize=12)
    axes[0].set_ylabel('Cluster', fontsize=12)
    axes[0].set_title('Cluster vs Grade (Counts)', fontsize=14, fontweight='bold')
    
    # Percentages with purity annotations
    sns.heatmap(ct_norm, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1],
                cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Grade', fontsize=12)
    axes[1].set_ylabel('Cluster', fontsize=12)
    axes[1].set_title('Cluster vs Grade (% within cluster)', fontsize=14, fontweight='bold')
    
    # Add purity annotations on the right side
    for i, (cluster_id, purity) in enumerate(cluster_purities.items()):
        axes[1].text(len(ct.columns) + 0.3, i + 0.5, f'Purity: {purity:.1f}%', 
                     va='center', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    plt.suptitle(config_label, fontsize=12, y=1.00)
    plt.tight_layout()
    
    filename = f'clusters/confusion_{config_label}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def create_summary_report(results_df, n_graded_samples):
    """Create summary visualizations comparing all methods"""
    
    # 1. Best methods by metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 15 by Purity (THIS IS WHAT MATTERS)
    top_purity = results_df.nlargest(15, 'purity')
    axes[0, 0].barh(range(len(top_purity)), top_purity['purity'], color='#2ecc71')
    axes[0, 0].set_yticks(range(len(top_purity)))
    axes[0, 0].set_yticklabels(top_purity['config'], fontsize=8)
    axes[0, 0].set_xlabel('Cluster Purity (Grade Agreement)')
    axes[0, 0].set_title(f'Top 15 by Purity - CLUSTER-GRADE AGREEMENT\n(evaluated on {n_graded_samples} graded samples)', 
                         fontweight='bold', fontsize=12)
    axes[0, 0].invert_yaxis()
    
    # Top 15 by Silhouette
    top_silhouette = results_df.nlargest(15, 'silhouette')
    axes[0, 1].barh(range(len(top_silhouette)), top_silhouette['silhouette'], color='steelblue')
    axes[0, 1].set_yticks(range(len(top_silhouette)))
    axes[0, 1].set_yticklabels(top_silhouette['config'], fontsize=8)
    axes[0, 1].set_xlabel('Silhouette Score')
    axes[0, 1].set_title('Top 15 Configurations by Silhouette Score', fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # Top 15 by ARI
    top_ari = results_df.nlargest(15, 'ari_overall')
    axes[1, 0].barh(range(len(top_ari)), top_ari['ari_overall'], color='coral')
    axes[1, 0].set_yticks(range(len(top_ari)))
    axes[1, 0].set_yticklabels(top_ari['config'], fontsize=8)
    axes[1, 0].set_xlabel('Adjusted Rand Index')
    axes[1, 0].set_title('Top 15 Configurations by ARI (vs Grades)', fontweight='bold')
    axes[1, 0].invert_yaxis()
    
    # Scatter: Purity vs Silhouette (colored by n_clusters)
    scatter = axes[1, 1].scatter(results_df['purity'], results_df['silhouette'], 
                                  c=results_df['n_clusters'], cmap='viridis', 
                                  s=100, alpha=0.6, edgecolors='black')
    axes[1, 1].set_xlabel('Purity (Grade Agreement)')
    axes[1, 1].set_ylabel('Silhouette Score')
    axes[1, 1].set_title('Purity vs Silhouette (colored by n_clusters)', fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 1], label='Number of Clusters')
    
    plt.tight_layout()
    plt.savefig('clusters/summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by method type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # By feature selection method
    feat_grouped = results_df.groupby('feature_method')[['purity', 'silhouette', 'ari_overall']].mean()
    feat_grouped['purity'].plot(kind='barh', ax=axes[0, 0], color='#2ecc71')
    axes[0, 0].set_xlabel('Mean Purity Score')
    axes[0, 0].set_title('Purity by Feature Selection Method', fontweight='bold')
    
    # By PCA method
    pca_grouped = results_df.groupby('pca_method')[['purity', 'silhouette', 'ari_overall']].mean()
    pca_grouped['purity'].plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_xlabel('Mean Purity Score')
    axes[0, 1].set_title('Purity by PCA Method', fontweight='bold')
    
    # By number of clusters
    cluster_grouped = results_df.groupby('n_clusters')[['purity', 'silhouette', 'ari_overall']].mean()
    cluster_grouped[['purity', 'silhouette', 'ari_overall']].plot(kind='line', ax=axes[1, 0], marker='o')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance by Number of Clusters', fontweight='bold')
    axes[1, 0].legend(['Purity', 'Silhouette', 'ARI'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of purity scores
    axes[1, 1].hist(results_df['purity'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(results_df['purity'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {results_df["purity"].mean():.3f}')
    axes[1, 1].set_xlabel('Purity Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Purity Scores', fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('clusters/summary_by_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSummary statistics saved to clusters/summary_*.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python clustering_analysis.py <model_name> <grade_column>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    grade = sys.argv[2]
    main(model_name, grade)
