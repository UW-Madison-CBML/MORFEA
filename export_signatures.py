import tphate
import pandas as pd
import numpy as np 
import scipy
from scipy.interpolate import make_interp_spline
import itertools
import os
# ig assume embryo timesteps are equally spaced
def get_quad_tphate_interp(latents):
    tphate_op = tphate.TPHATE(n_jobs=n_jobs, n_components=3)
    tphate_data = tphate_op.fit_transform(latents) 
    timesteps = np.linspace(0, 1, tphate_data.shape[0])
    x_data = tphate_data[:, 0]
    y_data = tphate_data[:, 1]
    z_data = tphate_data[:, 2]

    interp_x = make_interp_spline(timesteps, x_data, k=2, bc_type='natural')
    interp_y = make_interp_spline(timesteps, y_data, k=2, bc_type='natural')
    interp_z = make_interp_spline(timesteps, z_data, k=2, bc_type='natural')

    def interpolated_function(t_new):
        """Evaluates the interpolated 3D coordinates at new timesteps."""
        t_new = np.array(t_new)
        # Evaluate each component function and stack them into a (m, 3) array
        new_x = interp_x(t_new)
        new_y = interp_y(t_new)
        new_z = interp_z(t_new)
        return np.stack([new_x, new_y, new_z], axis=-1)

    return interpolated_function

def compute_path_signature(X, a=0, b=1, level_threshold=3):
    N = len(X)
    t = np.linspace(a, b, 10**4)
    dt = t[1] - t[0]
    X_t = [Xi(t) for Xi in X]
    t = t[:-1]
    dX_t = [np.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    
    signature = [[np.ones(len(t))]]
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(np.cumsum(previous_level_integral * dX_t[i]))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    
    return t, X_t, X_prime_t, signature, signature_terms
def main(model_name):
    for root, dirs, files in os.walk("."):
        level = root.replace(".", '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}[{os.path.basename(root)}/]')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')
    file_name = "latents/"+ model_name
    #file_name =
    keys = pd.read_csv(file_name+".csv")
    values = np.load(file_name+'.npy')
    if(len(keys) != values.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = ["z_{i}" for i in range(values.shape[1])]
    values_df = pd.DataFrame(values, columns=lat_columns)
    df = pd.concat([keys, values_df], axis = 0)
    df.groupby('cell_id').apply(lambda x: print(compute_path_signature(get_quad_tphate_interp(x[lat_columns].to_numpy()))))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
