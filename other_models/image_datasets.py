import torch
from torch.utils.data import Dataset
import pandas as pd
from dataset_ivf_embryo import read_gray, normalize_video
import torch
import os
class ImageGradeDataset(Dataset):
    GRADES = ["A","B","C"]
    def __init__(self, index_df, grade_df, grade="TE"):
        # index_df cols = embryo_id, embryo_paths
        # grade_df cols = embryo_id, TE, ICM, keep_default_na=False
        # grade = "TE" or "ICM" 
        index_df = index_df.merge(grade_df, how="left", left_on="embryo_id", right_on="embryo_id")
        index_df = index_df.dropna(subset=[grade])
        dfs = [] 
        for idx, row in index_df.iterrows():
            # embryo_id, num_frames, embryo_paths
            df = pd.DataFrame({"path":row["embryo_paths"].split("|"), "embryo_id": row["embryo_id"], grade:row[grade]})
            dfs.append(df)
        self.df = pd.concat(dfs, axis=0, ignore_index=True)
        self.grade = grade
        
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.tensor(normalize_video([read_gray(row["path"], 128, 50)], "minmax01")[0]), torch.tensor(self.__class__.GRADES.index(row[self.grade]), dtype=torch.long)
        
    def __len__(self):
        return len(self.df)



class ImageStageDataset(Dataset):
    PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
    @staticmethod
    def add_group_annotations(group, annotations_dir):
        annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
        df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

        new_column = []
        
        new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
        col_len_seq = []
        for index, row in df.iterrows():
            new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
            col_len_seq.append(len(new_column))


        new_column += ["post_phase"] * (len(group) - len(new_column))
        new_column = new_column[:len(group)]
        group["phase"] = new_column
        return group
     
        
        
    def __init__(self, index_df, annotations_dir):
        # index_df cols = embryo_id, embryo_paths
        # annotation_folder = path of the folder with stage annotations
        dfs = [] 
        for idx, row in index_df.iterrows():
            # embryo_id,num_frames,embryo_paths
            df = pd.DataFrame({"path":row["embryo_paths"].split("|"), "embryo_id": row["embryo_id"], "TE":row["TE"], "ICM":row["ICM"]})
            dfs.append(df)
        expanded_df = pd.concat(dfs, axis=0, ignore_index=True)
        self.df = expanded_df.groupby("embryo_id", group_keys=False).apply(lambda group: self.__class__.add_group_annotations(group, annotations_dir)).reset_index()
        

        
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.tensor(normalize_video([read_gray(row["path"], 128)], "minmax01")[0]), torch.tensor(self.__class__.index(row["phase"]), dtype=torch.long)
        
    def __len__(self):
        return len(self.df)



        
