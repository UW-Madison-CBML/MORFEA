import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
GRADE_COLORS = {"A":(0.6,1.0,0.6), "B":(1.0,1.0,0.6),"C":(1.0,0.6,0.6)}
def main(model_name):

    metadata_df = pd.read_csv(os.path.join("kanakasabapathy_latents",f"{model_name}.csv"))
    latents = np.load(os.path.join("kanakasabapathy_latents",f"{model_name}.npy"))
    pca_latents = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(latents))
    colors = [GRADE_COLORS[g] for g in metadata_df["TE"].to_list()]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(pca_latents[:,0], pca_latents[:,1], c=colors)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    fig.savefig("single_frame_plot.svg")
    plt.close(fig)


    
if __name__ == "__main__":
    import sys
    main(sys.argv[1])
