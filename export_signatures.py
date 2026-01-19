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
def get_quad_tphate_interp(latents):
    #tphate_op = tphate.TPHATE(n_jobs=8, n_components=3)
    #tphate_data = tphate_op.fit_transform(latents) 
    timesteps = np.linspace(0, 1, latents.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latents)

    # 3. Apply PCA
    # Since the input has 2 dimensions (features), we can set n_components=2 
    # to get all principal components, or n_components=1 to reduce dimensionality to 1D.
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    # 4. Transform the data
    X_pca = pca.transform(X_scaled)

    #x_data = tphate_data[:, 0]
    #y_data = tphate_data[:, 1]
    #z_data = tphate_data[:, 2]

    #interp_x = make_interp_spline(timesteps, x_data, k=2)
    #interp_y = make_interp_spline(timesteps, y_data, k=2)
    #interp_z = make_interp_spline(timesteps, z_data, k=2)

    interps = [make_interp_spline(timesteps, X_pca[:,i], k=2) for i in range(X_pca.shape[1])]

    return interps


def compute_path_signature(X, a=0, b=1, level_threshold=3, n_points=1000):
    """
    Compute path signature.

    Args:
        n_points: Number of discretization points (default: 1000)
                  Original was 10000 which is very slow!
                  1000 gives ~10x speedup with minimal accuracy loss
    """
    N = len(X)
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
                sig_flat.append(np.cumsum(previous_level_integral * dX_t[i]))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    print(len(sig_flat))    
    return t, X_t, X_prime_t, signature, signature_terms, np.array(sig_flat)
def get_new_row(group, cell_id):
    (_,_,_,sig,terms, signature) = compute_path_signature(get_quad_tphate_interp(group))
    sig = flatten_list(sig)
    terms = flatten_list(terms)
    print(signature.shape)
    print(len(sig), " ", terms)
    signature = signature.reshape(-1)
    # Return a Series instead of DataFrame for proper groupby handling
    result = pd.Series({'cell_id': cell_id})
    for i, val in enumerate(signature):
        result[f"s_{i}"] = val
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
    # This returns a DataFrame where each row is a cell_id with its signature
    signatures_df = df.groupby('embryo_id').apply(
        lambda group: get_new_row(group[lat_columns].to_numpy(), group.name)
    ).reset_index(drop=True)

    # Save to CSV
    signatures_df.to_csv("signatures/" + model_name + "_sigs.csv", index=False)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
