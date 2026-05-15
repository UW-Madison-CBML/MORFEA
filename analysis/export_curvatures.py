import tphate
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline
import itertools
import os
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from scipy.stats import kurtosis
def fit_circle_curvature(points, how="triangle"):
    """
    Fit a circle to 3 consecutive points and return curvature (1/radius).
    If points are collinear or too close, return 0.
    """
    if(how == "triangle"):
    
        # Get three points
        points = points[::max(1,len(points)//3)]
        p1, p2, p3 = points[0], points[1], points[2]

        # Calculate the radius using the circumradius formula
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Area using Heron's formula
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        
        if area_squared <= 0:
            return 0  # Collinear points
        
        area = np.sqrt(area_squared)
         
        if area == 0:
            return 0
        
        # Radius = (a*b*c) / (4*Area)
        radius = (a * b * c) / (4 * area)
        if radius == 0:
            return 0
        return 1 / radius
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(points)
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        points_2d = pca.transform(X_scaled)       
        def circle_residuals(params, points):
            xc, yc, R = params
            # Calculate distance from each point to the center (xc, yc)
            distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
            # The residual is the difference between these distances and the radius R
            return distances - R
        x = points[:,0]
        y = points[:,1]
        points = np.column_stack((x, y))

        x0 = [np.mean(x), np.mean(y), np.std(x)]

        res = least_squares(circle_residuals, x0, args=(points,))

        _, _, radius = res.x
        
        if(radius == 0):
            return 0 
        return 1/radius
 
def calculate_curvatures(trajectory):
    """Calculate curvature for each point in trajectory using sliding window."""
    offset = 6
    curvatures = []
    
    for i in range(len(trajectory)):
        if i < offset:
            points = trajectory[i:i+(2*offset)]
            curvatures.append(fit_circle_curvature(points))
        elif i >= len(trajectory) - offset:
            points = trajectory[i-(2*offset):i]
            curvatures.append(fit_circle_curvature(points))
        else:
            points = trajectory[i-offset:i+offset]
            curvatures.append(fit_circle_curvature(points))
    
    return np.array(curvatures)
def get_new_row(group, cell_id):
    if(np.isnan(group).any()):
        print(f"{cell_id} has nan!!!")
    curvatures = np.array(calculate_curvatures(group))
    result = pd.DataFrame({"cell_id":[cell_id]*len(group), "curvature":curvatures})
    
    return result

def main(model_name):
    file_name = "latents/"+ model_name
    #file_name =
    keys = pd.read_csv(file_name+".csv")
    values = np.load(file_name+'.npy')
    if(len(keys) != values.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(values.shape[1])]
    values_df = pd.DataFrame(values, columns=lat_columns)
    df = pd.concat([keys, values_df], axis = 1)
    sizes = df.groupby("embryo_id")["time_step"].size()
    print(sizes.idxmax())
    max_points = sizes.max()
    print("max points", max_points)
    signatures_df = df.groupby('embryo_id').apply(
        lambda group: get_new_row(group[lat_columns].to_numpy(), group.name)
    ).reset_index(drop=True)

    # Save to CSV
    signatures_df.to_csv("curvatures/" + model_name + ".csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
