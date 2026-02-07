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
    df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", "TE"] + sig_cols].dropna(subset=["TE"])
    feature_cols = [col for col in df.columns if col.startswith('s_')]
    X = df[feature_cols].values
    grades = df['TE'].values

    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features")
    print(f"Grade distribution: {dict(pd.Series(grades).value_counts().sort_index())}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n=== DIAGNOSTIC ANALYSIS ===\n")

    # 1. Feature statistics by grade
    print("Feature statistics by grade:")
    for grade in sorted(set(grades)):
        mask = grades == grade
        grade_features = X[mask]
        print(f"\nGrade {grade} (n={sum(mask)}):")
        print(f"  Mean: {grade_features.mean():.4f}")
        print(f"  Std:  {grade_features.std():.4f}")
        print(f"  Min:  {grade_features.min():.4f}")
        print(f"  Max:  {grade_features.max():.4f}")

    # 2. Can supervised learning predict grades?
    print("\n\nCan we predict grades from features?")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_scaled, grades, cv=5)
    baseline_accuracy = max(pd.Series(grades).value_counts()) / len(grades)
    print(f"Random Forest CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Baseline (always predict majority): {baseline_accuracy:.3f}")
    print(f"Improvement: {scores.mean() - baseline_accuracy:.3f}")
    if scores.mean() - baseline_accuracy < 0.05:
        print("⚠️  Features barely predict grades better than guessing!")

    # 3. Feature importance
    rf.fit(X_scaled, grades)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 most important features:")
    print(importances.head(10).to_string(index=False))

    # 4. Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n\nPCA: PC1 explains {pca.explained_variance_ratio_[0]:.1%}, PC2 explains {pca.explained_variance_ratio_[1]:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By grade
    grade_colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    colors = [grade_colors[g] for g in grades]
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=20, alpha=0.5, edgecolors='none')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Colored by Grade (A=green, B=orange, C=red)')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='A'), 
                       Patch(facecolor='orange', label='B'),
                       Patch(facecolor='red', label='C')]
    axes[0].legend(handles=legend_elements)

    # By cluster
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='average')
    clusters = hierarchical.fit_predict(X_scaled)
    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=20, alpha=0.5, edgecolors='none')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('Unsupervised Clustering (3 groups)')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    plt.tight_layout()
    plt.savefig('diagnostic_plot.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'diagnostic_plot.png'")

    print("\n=== SUMMARY ===")
    print("If the left plot (grades) shows no clear color separation,")
    print("and the right plot (clusters) doesn't match it,")
    print("then your features simply don't encode grade information.")
    print("Unsupervised clustering can't find what isn't there.")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)
