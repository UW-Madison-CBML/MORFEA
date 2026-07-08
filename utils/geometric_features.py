import numpy as np
import pandas as pd
import iisignature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
def get_path_sig(trajectory, depth, time_offsets = 1.0):
    # add time information
    trajectory = np.concatenate([trajectory, time_offsets * np.arange(trajectory.shape[0])[:,None]],axis=1)
    s_info = iisignature.prepare(trajectory.shape[1], depth)
    return iisignature.logsig(trajectory, s_info)

def get_path_sigs(trajectory, depth, time_offsets = 0.1, return_feature_labels=False):
    # add time information
    trajectory = np.concatenate([trajectory,time_offsets * np.arange(trajectory.shape[0])[:,None]],axis=1)
    s_info = iisignature.prepare(trajectory.shape[1], depth)
    signature = []
    for i in range(len(trajectory)):
        signature.append(iisignature.logsig(trajectory[:i+1], s_info))
    if(return_feature_labels):
        return np.array(signature),  iisignature.basis(s_info)
    else:
        return np.array(signature)

def fit_circle_curvature(points, how=""):
    if(how == "triangle"):
    
        # Get three points
        p1, p2, p3 = points[0], points[len(points)//2], points[-1]

        # Calculate the radius using the circumradius formula
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Area using Heron's formula
        s = (a + b + c) / 2
        area_squared = 0.0
        try:
            area_squared = s * (s - a) * (s - b) * (s - c)
        except OverflowError: 
            return 0
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
 
def calculate_curvatures(trajectory, offset = 10, how="triangle", retrospective=False):
    curvatures = []
    
    for i in range(len(trajectory)):
        points = trajectory[max(0,i-offset):i] if retrospective else trajectory[max(0,i-(offset//2)):min(len(trajectory), i+(offset//2))]
        if len(points) <= 3:
            curvatures.append(0)
        else:
            curvatures.append(fit_circle_curvature(points, how=how))
    
    return np.array(curvatures)

def get_dynam_features(trajectory):
    features = pd.DataFrame()

    return features
def get_vel(trajectory, dt=1):
    v_vecs = np.diff(trajectory, axis=0) / dt
    v_scalars = np.linalg.norm(v_vecs, axis=1)

    v_padded = np.pad(v_scalars, (0, 1), mode='edge')

    return v_padded

def get_acc(trajectory, dt=1):
    v_vecs = np.diff(trajectory, axis=0) / dt
    v_scalars = np.linalg.norm(v_vecs, axis=1)

    v_padded = np.pad(v_scalars, (0, 1), mode='edge')

    a_vecs = np.diff(v_vecs, axis=0) / dt
    a_scalars = np.linalg.norm(a_vecs, axis=1)

    a_padded = np.pad(a_scalars, (1, 1), mode='edge')
    return a_padded


