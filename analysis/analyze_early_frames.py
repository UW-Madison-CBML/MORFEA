import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
#from matplotlib.colormaps import Accent
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
def main():

    pca_df = pd.read_csv(os.path.abspath("pca_val.csv"))
    X = pca_df[["pca_0","pca_1","pca_2"]].to_numpy()
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X) # fix the seed so we can rerun
    colors = [colormaps.get_cmap("Accent")(idx) for idx in kmeans.labels_]
    fig, ax = plt.subplots(figsize=(6,8), subplot_kw={"projection":"3d"})
    ax.scatter(X[:,0], X[:,1], X[:,2], c= colors)
    legend_elements = [Patch(facecolor=colormaps.get_cmap("Accent")(i), label=i) for i in range(8)]
    fig.legend(handles=legend_elements, title="Clusters") 
   
    fig.savefig("pca_clusters.png")
    
    plt.close(fig)    
    print(pca_df[kmeans.labels_ == 1]) 

    print(pca_df[kmeans.labels_ == 3]) 
    print(pca_df["embryo_id"].unique())
    
if __name__ == "__main__":
    main()
