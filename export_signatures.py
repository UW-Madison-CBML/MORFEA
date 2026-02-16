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


def compute_path_signature(X, a=0, b=1, level_threshold=3, n_points=1000):
    """
    Compute path signature.

    Args:
        n_points: Number of discretization points (default: 1000)
                  Original was 10000 which is very slow!
                  1000 gives ~10x speedup with minimal accuracy loss
    """
    
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


    # Kurtosis: fourth central moment (fisher=True means Normal = 0)
    #new_rows = []
    #for i in range(50):
    #interped_latents = np.array([i(np.linspace(0,1,500)) for i in get_quad_tphate_interp(group, how="FULL", n_components=10)]).T if max_len == 0 else group
    #if(np.isnan(interped_latents).any()):
    #    print(f"{cell_id} has nan!!!")
    #signature = np.array(calculate_curvatures(interped_latents))
    trajectory = group
    # Basic velocity
    velocity = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    features = {}
    # === DISTRIBUTION STATISTICS ===
    features['velocity_mean'] = np.mean(velocity)
    features['velocity_std'] = np.std(velocity)
    features['velocity_median'] = np.median(velocity)
    features['velocity_min'] = np.min(velocity)
    features['velocity_max'] = np.max(velocity)
    features['velocity_range'] = np.max(velocity) - np.min(velocity)

    # Percentiles (capture outliers)
    features['velocity_p10'] = np.percentile(velocity, 10)
    features['velocity_p25'] = np.percentile(velocity, 25)
    features['velocity_p75'] = np.percentile(velocity, 75)
    features['velocity_p90'] = np.percentile(velocity, 90)
    features['velocity_iqr'] = np.percentile(velocity, 75) - np.percentile(velocity, 25)

    # Shape of distribution
    from scipy.stats import skew, kurtosis
    features['velocity_skewness'] = skew(velocity)
    features['velocity_kurtosis'] = kurtosis(velocity)

    # === CONSISTENCY MEASURES ===
    # Coefficient of variation (normalized std)
    features['velocity_cv'] = np.std(velocity) / (np.mean(velocity) + 1e-10)

    # What % of time is velocity "normal" (within 1 std of mean)?
    mean_v, std_v = np.mean(velocity), np.std(velocity)
    features['velocity_stability'] = np.sum((velocity > mean_v - std_v) & (velocity < mean_v + std_v)) / len(velocity)

    # How often does velocity change drastically?
    velocity_changes = np.abs(np.diff(velocity))
    features['velocity_change_mean'] = np.mean(velocity_changes)
    features['velocity_change_max'] = np.max(velocity_changes)

    # === TEMPORAL PATTERNS ===
    # Early vs late development speed
    third = len(velocity) // 3
    features['velocity_early'] = np.mean(velocity[:third])
    features['velocity_mid'] = np.mean(velocity[third:2*third])
    features['velocity_late'] = np.mean(velocity[2*third:])

    # Ratio of late to early (speeding up or slowing down?)
    features['velocity_late_early_ratio'] = features['velocity_late'] / (features['velocity_early'] + 1e-10)

    # Trend: is velocity increasing or decreasing over time?
    time_indices = np.arange(len(velocity))
    features['velocity_trend'] = np.polyfit(time_indices, velocity, 1)[0]  # slope of linear fit

    # === EXTREMES & OUTLIERS ===
    # Count of extremely fast/slow periods
    v_median = np.median(velocity)
    features['slow_periods'] = np.sum(velocity < 0.5 * v_median) / len(velocity)
    features['fast_periods'] = np.sum(velocity > 2.0 * v_median) / len(velocity)

    # Longest sustained high/low velocity
    high_velocity = velocity > features['velocity_p75']
    features['max_consecutive_high_vel'] = np.max(np.diff(np.where(np.concatenate(([high_velocity[0]], high_velocity[:-1] != high_velocity[1:], [True])))[0])[::2])

    low_velocity = velocity < features['velocity_p25']
    features['max_consecutive_low_vel'] = np.max(np.diff(np.where(np.concatenate(([low_velocity[0]], low_velocity[:-1] != low_velocity[1:], [True])))[0])[::2])

    # === MONOTONICITY ===
    # Is velocity mostly increasing/decreasing or all over the place?
    velocity_diffs = np.diff(velocity)
    features['velocity_increases'] = np.sum(velocity_diffs > 0) / len(velocity_diffs)
    features['velocity_decreases'] = np.sum(velocity_diffs < 0) / len(velocity_diffs)

    # === SMOOTHNESS ===
    # How smooth is the velocity curve? (2nd derivative of position)
    velocity_smoothness = np.std(np.diff(velocity))
    features['velocity_smoothness'] = velocity_smoothness

    # === PHASE TRANSITIONS (big jumps) ===
    # Detect developmental phase shifts
    velocity_jumps = np.abs(np.diff(velocity))
    features['n_phase_transitions'] = np.sum(velocity_jumps > 2 * np.std(velocity))
    features['biggest_phase_transition'] = np.max(velocity_jumps)
    # Basic acceleration
    acceleration = np.diff(velocity)

    # === DISTRIBUTION STATISTICS ===
    features['accel_mean'] = np.mean(acceleration)
    features['accel_std'] = np.std(acceleration)
    features['accel_median'] = np.median(acceleration)
    features['accel_mean_abs'] = np.mean(np.abs(acceleration))  # Magnitude regardless of direction
    features['accel_max_positive'] = np.max(acceleration)
    features['accel_max_negative'] = np.min(acceleration)
    features['accel_range'] = np.max(acceleration) - np.min(acceleration)

    # Percentiles
    features['accel_p10'] = np.percentile(acceleration, 10)
    features['accel_p90'] = np.percentile(acceleration, 90)

    # Distribution shape
    features['accel_skewness'] = skew(acceleration)
    features['accel_kurtosis'] = kurtosis(acceleration)

    # === DIRECTION CHANGES ===
    # How often does acceleration switch from positive to negative?
    accel_sign_changes = np.sum(np.diff(np.sign(acceleration)) != 0)
    features['accel_sign_changes'] = accel_sign_changes
    features['accel_sign_change_rate'] = accel_sign_changes / len(acceleration)

    # Ratio of acceleration vs deceleration
    features['accel_positive_ratio'] = np.sum(acceleration > 0) / len(acceleration)
    features['accel_negative_ratio'] = np.sum(acceleration < 0) / len(acceleration)

    # === TEMPORAL PATTERNS ===
    # Early vs late acceleration patterns
    third = len(acceleration) // 3
    features['accel_early'] = np.mean(acceleration[:third])
    features['accel_mid'] = np.mean(acceleration[third:2*third])
    features['accel_late'] = np.mean(acceleration[2*third:])

    # Is the embryo accelerating or decelerating overall?
    features['accel_trend'] = np.polyfit(np.arange(len(acceleration)), acceleration, 1)[0]

    # === SUSTAINED PERIODS ===
    # Longest period of sustained acceleration
    sustained_accel = acceleration > 0
    if np.any(sustained_accel):
        features['max_sustained_accel'] = np.max(np.diff(np.where(np.concatenate(([sustained_accel[0]], sustained_accel[:-1] != sustained_accel[1:], [True])))[0])[::2])
    else:
        features['max_sustained_accel'] = 0

    # Longest period of sustained deceleration
    sustained_decel = acceleration < 0
    if np.any(sustained_decel):
        features['max_sustained_decel'] = np.max(np.diff(np.where(np.concatenate(([sustained_decel[0]], sustained_decel[:-1] != sustained_decel[1:], [True])))[0])[::2])
    else:
        features['max_sustained_decel'] = 0

    # === EXTREMES ===
    # Strong acceleration/deceleration events
    accel_threshold = 2 * np.std(acceleration)
    features['strong_accel_events'] = np.sum(acceleration > accel_threshold)
    features['strong_decel_events'] = np.sum(acceleration < -accel_threshold)

    # === VARIABILITY ===
    # Coefficient of variation
    features['accel_cv'] = np.std(acceleration) / (np.mean(np.abs(acceleration)) + 1e-10)

    # === CUMULATIVE MEASURES ===
    # Total acceleration applied over trajectory
    features['total_accel'] = np.sum(np.abs(acceleration))
    features['net_accel'] = np.sum(acceleration)  # Can be negative

    # === CONSISTENCY ===
    # How "jerky" is development? (variance of acceleration changes)
    features['accel_jerkiness'] = np.std(np.diff(acceleration))

    # Rolling window variability (local instability)
    window_size = max(3, len(acceleration) // 10)
    rolling_std = []
    for i in range(len(acceleration) - window_size):
        rolling_std.append(np.std(acceleration[i:i+window_size]))
    features['accel_local_variability'] = np.mean(rolling_std) if rolling_std else 0 
    if max_len > 0:
        if(len(signature) > max_len):
            raise ValueError(f"{cell_id} exceeds max len for padding")
        elif(len(signature) != max_len):
            pad_width = max(0, max_len - len(signature))
            #signature = np.pad(signature, ((0, pad_width), (0, 0)), mode='constant') 
            signature = np.pad(signature, (0, pad_width), mode='constant') 
    signature = np.array([val for val in features.values()])
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
        lambda group: get_new_row(group[lat_columns].to_numpy(), group.name)#, max_len=max_points)
    ).reset_index(drop=True)

    # Save to CSV
    signatures_df.to_csv("signatures/" + model_name + "_sigs.csv", index=False)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()
    
    main(args.name)
