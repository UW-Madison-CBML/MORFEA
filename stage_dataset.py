import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
 
def calculate_curvatures(group, offset):

    """Calculate curvature for each point in trajectory using sliding window."""

    lat_cols = [column for column in group.columns if column.startswith("z_")]

    trajectory = group[lat_cols].values
    curvatures = []
 
    for i in range(len(trajectory)):

        points = trajectory[max(0,i-offset):i]
        if len(points) < 3:
            curvatures.append(0) 
        else:
            curvatures.append(fit_circle_curvature(points))
 
    return np.array(curvatures)
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial import distance_matrix
def addAnnotations(group_name, group, annotations_dir, curvature = True, velocity = True, latents = True, acceleration = True, path_signatures = True, distance_mat=True):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (len(group) - len(new_column))
    new_column = new_column[:len(group)]
    
    group["phase"] = new_column

    trajectory = group[lat_cols].values

    if (curvature):
        group["z_curvature_12"] = calculate_curvatures(group, 12)
        group["z_curvature_12"] = group["z_curvature_12"] * (1 / (group["z_curvature_12"].std() + 0.0001))
        group["z_curvature_20"] = calculate_curvatures(group, 20)
        group["z_curvature_20"] = group["z_curvature_20"] * (1 / (group["z_curvature_20"].std() + 0.0001))
        group["z_curvature_4"] = calculate_curvatures(group, 4)
        group["z_curvature_4"] = group["z_curvature_4"] * (1 / (group["z_curvature_4"].std() + 0.0001))


    if (path_signatures != None):
        print("")
    if(distance_mat):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if(acceleration):
        group["z_speed"] = group[lat_cols].diff(axis=0).mean(axis=1).fillna(0)
        group["z_acc"] = group[lat_cols].diff(axis=0).diff(axis=0).mean(axis=1).fillna(0)
         


    if (not latents):
        group = group.drop(columns=lat_cols)
    
    
    return group

class StageDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, annotations_dir, curvature=True, velocity=True, latents=True, acceleration=True, path_signatures=True, return_embryo_id=False, distance_mat=True): # preparing latents_df outside of the class i.e. from .csv .npy in latents/
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.latents_df = latents_df
        self.annotations_dir = annotations_dir
        self.return_embryo_id = return_embryo_id
        sizes = self.latents_df.groupby("embryo_id")["time_step"].size()
        self.max_points = sizes.max()
        
        self.df = self.latents_df.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,self.annotations_dir, curvature = curvature, velocity = velocity, latents = latents, acceleration = acceleration, path_signatures = None, distance_mat=distance_mat)).reset_index()
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64
        

        self.phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
        self.lat_cols = [column for column in self.df.columns if column.startswith("z_")]
        # 1. dim reduce latents_df
        # dir/EMBRYO_ID_annotions.csv
        #   stage_id, stage_begin, stage_end
        #   stage1, 10, 50
        #   stage2, 51, 70
        # 2. pick features (velocity, curvature, etc.) and labels (stage)
        # 3. build self.df

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        group = self.groups.get_group(row["embryo_id"])

        
        seqindex = int(((row["time_step"] - 1) / len(group)) * (len(group) - self.seqlength - 1))

        seq_df = group.iloc[seqindex : seqindex + self.seqlength]

        if (self.return_embryo_id):

            return seq_df[self.lat_cols].to_numpy(), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long), row["embryo_id"]

        return seq_df[self.lat_cols].to_numpy(), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long)
        

        # [EMBRYO1, 0, .... (latent vector),
        # EMBRYO1, 1, .... (latent vector)]
        # [EMBRYO2, 0, .... (latent vector),
        # EMBRYO2, 1, .... (latent vector)]
        # grab index with self.df.loc[idx]
        # that row['stage'] will be in some set of ["stage0","stage1","stage2"...],
        # CEL does not expect [0,0,1...] but rather 1 "stage1", 3 "stage3"
        # return velocity, curvature, stage (stage is an long 64 integer)


    def __len__(self):
        return len(self.df)
