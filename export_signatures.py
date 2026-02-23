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

from scipy.stats import skew, kurtosis
from scipy.optimize import least_squares
from scipy.stats import kurtosis
def fit_circle_curvature(points, how=""):
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
        if(np.isnan(points).any()):
            return 0 
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
        x = points_2d[:,0]
        y = points_2d[:,1]
        points = np.column_stack((x, y))

        x0 = [np.mean(x), np.mean(y), np.std(np.sqrt(x**2 + y**2))]
        try:
            res = least_squares(circle_residuals, x0, args=(points,))
        except ValueError as e:
            print(e)
            return 0
        if not res.success:
            return 0
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


def flatten_list(nested_list):
    """
    Recursively flattens an arbitrarily nested list.
    """
    flat_list = []
    for item in nested_list:
        # Check if the item is a list (but not a string, which is also iterable)
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
# ig assume embryo timesteps are equally spaced
def get_quad_tphate_interp(latents, how="PCA", n_components=2):
    X_out = np.zeros((latents.shape[0], n_components))
    if(how == "TPHATE"):
        tphate_op = tphate.TPHATE(n_jobs=8, n_components=n_components)
        X_out = tphate_op.fit_transform(latents) 

    elif(how == "PCA"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(latents)

        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        X_out = pca.transform(X_scaled)
    elif(how == "UMAP"):
        mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)

        X_out = mapper.fit_transform(latents)
    elif(how == "FULL"):
        X_out = latents 
    else:
        raise ValueError("invalid how")
    timesteps = np.linspace(0, 1, latents.shape[0])
    interps = [make_interp_spline(timesteps, X_out[:,i], k=3) for i in range(X_out.shape[1])]

    return interps
def get_accel_features(velocity):
    features['velocity_mean'] = np.mean(velocity)
    features['velocity_std'] = np.std(velocity)
    features['velocity_median'] = np.median(velocity)
    features['velocity_min'] = np.min(velocity)
    features['velocity_max'] = np.max(velocity)
    features['velocity_range'] = np.max(velocity) - np.min(velocity)

    features['velocity_p10'] = np.percentile(velocity, 10)
    features['velocity_p25'] = np.percentile(velocity, 25)
    features['velocity_p75'] = np.percentile(velocity, 75)
    features['velocity_p90'] = np.percentile(velocity, 90)
    features['velocity_iqr'] = np.percentile(velocity, 75) - np.percentile(velocity, 25)

    features['velocity_skewness'] = skew(velocity)
    features['velocity_kurtosis'] = kurtosis(velocity)
    return features
     

def compute_path_signature(X, a=0, b=1, level_threshold=3, n_points=1000):

    N = len(X)
    level_threshold=N
    t = np.linspace(a, b, n_points)
    dt = t[1] - t[0]
    X_t = [Xi(t) for Xi in X]
    t = t[:-1]
    dX_t = [np.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    sig_flat = [] 
    signature = [[np.ones(len(t))]]
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(np.cumsum(previous_level_integral * dX_t[i]))
                sig_flat.append(np.cumsum(previous_level_integral * dX_t[i])[-1])
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    return t, X_t, X_prime_t, signature, signature_terms, np.array(sig_flat)
def get_new_row(group, cell_id, max_len=0):
    #(_,_,_,sig,terms, signature) = compute_path_signature(get_quad_tphate_interp(group, how="FULL", n_components=0))
    #signature = np.array([i(np.linspace(0,1, 500)) for i in get_quad_tphate_interp(group, how="FULL", n_components=0)]).flatten()
    #signature = group[:-50,:].flatten()


    #new_rows = []
    #for i in range(50):
    interped_latents = np.array([i(np.linspace(0,1,500)) for i in get_quad_tphate_interp(group, how="FULL", n_components=10)]).T if max_len == 0 else group
    #if(np.isnan(interped_latents).any()):
    #    print(f"{cell_id} has nan!!!")
    curvature = np.array(calculate_curvatures(interped_latents))
    trajectory = group
    # Basic velocity
    #velocity = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    #features = get_accel_features(velocity)
    
    if max_len > 0:
        if(len(signature) > max_len):
            raise ValueError(f"{cell_id} exceeds max len for padding")
        elif(len(signature) != max_len):
            pad_width = max(0, max_len - len(signature))
            #signature = np.pad(signature, ((0, pad_width), (0, 0)), mode='constant') 
            signature = np.pad(signature, (0, pad_width), mode='constant') 
    #signature = np.concatenate((np.array([val for val in features.values()]), np.array(curvature)))
    signature = np.array(curvature)
    #    #new_rows.append(signature)
    #    new_rows.append( 
    #new_rows = np.array(new_rows).T 
    #print(new_rows.shape)
    #sig = flatten_list(sig)
    #terms = flatten_list(terms)
    signature = signature.flatten()
    result = pd.DataFrame({"cell_id":[cell_id]})
    for i, val in enumerate(signature):
        result[f"s_{i}"] = val
    
    return result
def main(model_name):
    file_name = "latents/"+ model_name
    #file_name =
    keys = pd.read_csv(file_name+".csv").rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    values = np.load(file_name+'.npy')
    if(len(keys) != values.shape[0]):
        raise ValueError("keys and values sizes do not match")
    """scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)

    pca = PCA(n_components=100)
    pca.fit(X_scaled)

    values = pca.transform(X_scaled)""" 
    lat_columns = [f"z_{i}" for i in range(values.shape[1])]
    values_df = pd.DataFrame(values, columns=lat_columns)
    df = pd.concat([keys, values_df], axis = 1)
    # This returns a DataFrame where each row is a cell_id with its signature
    sizes = df.groupby("embryo_id")["time_step"].size()
    print(sizes.idxmax())
    max_points = sizes.max()
    print("max points", max_points)
    signatures_df = df.groupby('embryo_id').apply(
        lambda group: get_new_row(group[lat_columns].to_numpy().astype(np.float32), group.name, max_len=max_points)
    ).reset_index(drop=True)

    # Save to CSV
    signatures_df.to_csv("signatures/" + model_name + "_sigs.csv", index=False)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
