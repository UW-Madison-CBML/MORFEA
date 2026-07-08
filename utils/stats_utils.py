import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
def prfcm(gt_indices:torch.Tensor, pred_indices:torch.Tensor, num_classes):
    # gt_indicies1: shape = (B), 0 <= min(), max() < num_classes
    # pred_indicies2: shape = (B), 0 <= min(), max() < num_classes
    # 0 <= i < num_classes
    # returns: precision, recall, f1. shape = num_classes
    #          confusion_mat. shape = num_classes, num_classes

    confusion_mat = torch.einsum("bi, bj->ij", F.one_hot(gt_indices, num_classes=num_classes), F.one_hot(pred_indices, num_classes=num_classes))
    diag = confusion_mat[torch.arange(num_classes),torch.arange(num_classes)]
    recall = torch.nan_to_num(diag/confusion_mat.sum(dim=0), 0.0)
    precision = torch.nan_to_num(diag/confusion_mat.sum(dim=1), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
    return (recall, precision, f1), confusion_mat

def disp_cm(cm:np.ndarray, labels):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 
    img = wandb.Image(fig)
    plt.close(fig)
    return img



