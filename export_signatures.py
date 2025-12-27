import tphate
import pandas as pd
import numpy as np 
import scipy
from scipy.interpolate import make_interp_spline
import itertools
import os
# ig assume embryo timesteps are equally spaced
def get_quad_tphate_interp(latents):
    tphate_op = tphate.TPHATE(n_jobs=1, n_components=3)
    tphate_data = tphate_op.fit_transform(latents) 
    timesteps = np.linspace(0, 1, tphate_data.shape[0])
    x_data = tphate_data[:, 0]
    y_data = tphate_data[:, 1]
    z_data = tphate_data[:, 2]

    interp_x = make_interp_spline(timesteps, x_data, k=2)
    interp_y = make_interp_spline(timesteps, y_data, k=2)
    interp_z = make_interp_spline(timesteps, z_data, k=2)

    return (interp_x, interp_y, interp_z)


def compute_path_signature(X, a=0, b=1, level_threshold=3):
    N = len(X)
    t = np.linspace(a, b, 10**4)
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
    
    return t, X_t, X_prime_t, signature, signature_terms, np.array(sig_flat)
def get_new_row(group, cell_id):
    (_,_,_,_,_, signature) = compute_path_signature(get_quad_tphate_interp(group))
    signature = signature.reshape(-1)
    cols = [f"s_{i}" for i in range(signature.shape[0])]
    signature_row = pd.DataFrame(signature.reshape(1, -1), columns=cols)
    cell_id_row = pd.DataFrame([cell_id], columns=["cell_id"])
    return pd.concat([cell_id_row, signature_row], axis = 0) 
def main(model_name):
    file_name = "latents/"+ model_name
    #file_name =
    keys = pd.read_csv(file_name+".csv")
    values = np.load(file_name+'.npy')
    if(len(keys) != values.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(values.shape[1])]
    values_df = pd.DataFrame(values, columns=lat_columns)
    df = pd.concat([keys, values_df], axis = 0)
    signatures_df = df.groupby('cell_id').apply(lambda group: get_new_row(group[lat_columns].to_numpy(), group.name))
    df.to_csv(model_name + "_sigs.csv")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
