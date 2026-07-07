import torch
from image_grade_dataset import ImageGradeDataset, SingleFrameDataset
from image_grade_model import ImageGradeModel, SingleFrameModel
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

def recall_precision_f1(confusion_mat, i):
    recall = 0 if confusion_mat[:, i].sum() == 0 else confusion_mat[i,i]/confusion_mat[:, i].sum()
            
    precision = 0 if confusion_mat[i,:].sum() == 0 else confusion_mat[i,i]/ confusion_mat[i, :].sum()
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1

VAL_EMBRYOS = [
    "RG434-11",
    "RC1103-1",
    "LV488-7",
    "QC211-6",
    "BM016-2",
    "LM184-3",
    "RMN410-3",
    "PA145-1",
    "RO793-2",
    "PV361-2",
    "RC755-7",
    "VC581-3",
    "VC581-11",
    "ADM715-1-2",
    "LS1045-4",
    "GA800-4",
    "GJ191-1",
    "JV227-2",
    "LA367-4",
    "BN356-6",
    "TN611-7",
    "AHS115-5",
    "LCF544-2",
    "JV227-5",
    "CAV074-8",
    "AL702-9",
    "VH99-3",
    "GE218-3",
    "CC455-3",
    "DA1054-5",
    "ME378-4",
    "BA560-1",
    "PA145-2",
    "DSM138-5",
    "FN852-1",
    "TJ297-4",
    "RC755-9",
    "PA289-8",
    "LS93-8",
    "GA817-1-8",
    "AM918-2-5",
    "LNA592-9",
    "RS363-7",
    "JE021-4",
    "ZS435-6",
    "QC211-2",
    "GE1055-6",
    "LBS371-1-8",
    "CAV074-1",
    "RA803-4",
    ]

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_dev(self):
        return math.sqrt(self.variance)

def train_single_frame_classifier():
    weights = [0.3,0.3,0.3]
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGE_SIZE = 224
    val_ratio = 0.2
    epochs = 500
    wandb.login(key=os.getenv("WANDB_KEY"))
    learning_rate = 0.005
    weight_decay = 1e-3
    run = wandb.init(
        name = "image_grade",
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={
            "lr":learning_rate,
            "image_size": IMAGE_SIZE,
            "epochs":epochs
                }
    )
    grade_options = ["A","B","C"]
    grade_df = pd.read_csv(os.path.abspath("embryo_dataset_grades.csv")).rename(columns={"video_name":"embryo_id"}) 
    
    images_3 = [os.path.join("kanakasabapathy","3",path) for path in os.listdir(os.path.join("kanakasabapathy","3"))]
    images_4 = [os.path.join("kanakasabapathy","4",path) for path in os.listdir(os.path.join("kanakasabapathy","4"))]
    images_5 = [os.path.join("kanakasabapathy","5",path) for path in os.listdir(os.path.join("kanakasabapathy","5"))]
    paths = images_3 + images_4 + images_5
    grades = (["C"] * len(images_3))+ (["B"] * len(images_4))+(["A"] * len(images_5))
    df = pd.DataFrame({"path":paths, "TE":grades, "embryo_id":np.arange(len(paths))}) #spoof the embryo id as just a number
    df = df.sample(frac=1, replace=False)
    VAL_RATIO = 0.2 
    val_index = int(VAL_RATIO * len(df))
    val_df = df.iloc[:val_index]
    df = df.iloc[val_index:]

    dataset = SingleFrameDataset(df, image_size=IMAGE_SIZE) 
    dataset_val = SingleFrameDataset(val_df, image_size=IMAGE_SIZE)
    crit = torch.nn.CrossEntropyLoss(weight= torch.tensor(weights, device=DEVICE))
    model = SingleFrameModel(image_size=IMAGE_SIZE)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True)
    loader_val = DataLoader(
        dataset_val,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False)

    scheduler = CosineAnnealingLR(optimizer, len(loader) * epochs)
    print(len(loader))
    for epoch in range(epochs):
        model.train()
        for features, targets in loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()
            label = model(features)
            loss = crit(label, targets)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()

            scheduler.step()
            run.log({"loss": loss.item()})
        acc_stats = RunningStats()

        model.eval()
        confusion_mat = torch.zeros((3,3))
        with torch.no_grad():
            for features, targets in loader_val:
                
                print(f"targets shape: {targets.shape}")

                print(f"features shape: {features.shape}")

                features = features.to(DEVICE)
                logits = model(features)
                
                preds = logits.cpu().argmax(dim=-1)
                
                acc_stats.push((preds == targets).sum().item()/targets.shape[0]) 
                
                targets = F.one_hot(targets, num_classes=3)
                preds = F.one_hot(preds, num_classes=3)


                confusion_mat += torch.einsum('ti,tj->ij', preds, targets) # this einsum calculates confusion mats


               
        val_dict = {} 
        for i, g in enumerate(grade_options):
            recall, precision, f1 = recall_precision_f1(confusion_mat, i)
            val_dict[f"{g}_recall"] = recall; val_dict[f"{g}_precision"] = precision; val_dict[f"{g}_f1"] = f1
            #---------------------------------------------------------------------------------- 
            
        run.log(val_dict | {"acc_mean": acc_stats.mean, "acc_std":acc_stats.std_dev })


def train_image_sequence_classifier():
    weights = [0.2,0.4,0.4]
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    torch.backends.cudnn.enabled = False

    val_ratio = 0.2
    wandb.login(key=os.getenv("WANDB_KEY"))
    learning_rate = 0.0001
    weight_decay = 1e-3
    run = wandb.init(
        name = "image_grade",
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={"lr":learning_rate}
    )
    grade_options = ["A","B","C"]
    embryo_image_df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})
    grade_df = pd.read_csv(os.path.abspath("embryo_dataset_grades.csv")).rename(columns={"video_name":"embryo_id"}) 
    df = embryo_image_df.merge(grade_df, how="left", left_on="embryo_id", right_on="embryo_id").dropna(subset=["TE"])
    
    te_graded = df.dropna(subset=["TE"])["embryo_id"].unique().tolist()
    print(te_graded)
    np.random.shuffle(te_graded)
    VAL_EMBRYOS = te_graded[:int(val_ratio * len(te_graded))] 
    print(VAL_EMBRYOS)
    mask = df["embryo_id"].isin(VAL_EMBRYOS)
    val_df = df[mask]
    df = df[~mask]
    
    dataset_te = ImageGradeDataset(df, "TE") 
    dataset_icm = ImageGradeDataset(df, "ICM")
    dataset_te_val = ImageGradeDataset(val_df, "TE", return_whole_seqs=True) 
    dataset_icm_val = ImageGradeDataset(val_df, "ICM", return_whole_seqs=True)
    crit_te = torch.nn.CrossEntropyLoss(weight= torch.tensor(weights, device=DEVICE))
    crit_icm = torch.nn.CrossEntropyLoss(weight= torch.tensor(weights, device=DEVICE))
    model_te = ImageGradeModel()
    model_te = model_te.to(DEVICE)
    model_icm = ImageGradeModel()
    model_icm = model_icm.to(DEVICE)
    epochs = 8 
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=learning_rate, weight_decay=weight_decay)


    loader_te = DataLoader(
        dataset_te,
        batch_size=16,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x:dataset_te.pad_collate(x))
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=16,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True, 
        collate_fn=lambda x:dataset_icm.pad_collate(x))
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda x:dataset_te_val.pad_collate(x))
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda x:dataset_icm_val.pad_collate(x))
    scheduler_te = CosineAnnealingLR(optimizer_te, len(loader_te) * epochs)
    scheduler_icm = CosineAnnealingLR(optimizer_icm, len(loader_icm) * epochs)

    print(len(loader_te))
    for epoch in range(epochs):
        model_te.train(); model_icm.train()
        # TE grades
        for features, targets, lengths in loader_te:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()
            label = model_te(features, lengths)
            loss = crit_te(label, targets)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()

            scheduler_te.step()
            run.log({"te": loss.item()})
        # ICM grades
        for features, targets, lengths in loader_icm:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()
            
            label = model_icm(features, lengths)
            loss = crit_icm(label, targets)

            optimizer_icm.zero_grad() 
            loss.backward() 
            optimizer_icm.step()
            scheduler_icm.step()
            run.log({"icm": loss.item()})
    
        te_acc_stats = RunningStats()
        icm_acc_stats = RunningStats()

        model_te.eval(); model_icm.eval()
        te_confusion_mat = torch.zeros((3,3))
        icm_confusion_mat = torch.zeros((3,3))
        with torch.no_grad():
            for features, targets, lengths in loader_te_val:
                
                print(f"targets shape: {targets.shape}")

                print(f"features shape: {features.shape}")

                features = features.to(DEVICE)
                logits = model_te(features, lengths)
                
                preds = logits.cpu().argmax(dim=-1)
                
                te_acc_stats.push((preds == targets).sum().item()/targets.shape[0]) 
                
                targets = F.one_hot(targets, num_classes=3)
                preds = F.one_hot(preds, num_classes=3)


                te_confusion_mat += torch.einsum('ti,tj->ij', preds, targets) # this einsum calculates confusion mats


            for features, targets, lengths in loader_icm_val:
                features = features.to(DEVICE)
                logits = model_icm(features, lengths)

                preds = logits.cpu().argmax(dim=-1) 

                icm_acc_stats.push((preds == targets).sum().item()/targets.shape[0])
                targets = F.one_hot(targets, num_classes=3)
                preds = F.one_hot(preds, num_classes=3)

                icm_confusion_mat += torch.einsum('ti,tj->ij', preds, targets)
                
        
        for i, g in enumerate(grade_options):
            recall, precision, f1 = recall_precision_f1(te_confusion_mat, i)
            run.log({f"te_{g}_recall": recall,f"te_{g}_precision": precision,f"te_{g}_f1":f1})
            #---------------------------------------------------------------------------------- 
            
            recall, precision, f1 = recall_precision_f1(icm_confusion_mat, i)
            run.log({f"icm_{g}_recall": recall,f"icm_{g}_precision": precision,f"icm_{g}_f1":f1})
        print(grade_options)
        print("te confusion:"); print(te_confusion_mat)
        print("icm confusion:"); print(icm_confusion_mat)
        run.log({"te_acc_mean": te_acc_stats.mean, "te_acc_std":te_acc_stats.std_dev, "icm_acc_mean":icm_acc_stats.mean, "icm_acc_std":icm_acc_stats.std_dev})

if __name__ == "__main__":
    import sys
    if(len(sys.argv) >= 2 and sys.argv[1] == "sequence"):
        train_image_sequence_classifier()
    else:
        train_single_frame_classifier()

