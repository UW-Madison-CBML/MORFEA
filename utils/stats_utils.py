import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
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


def prfcm(gt_indices:torch.Tensor, pred_indices:torch.Tensor, num_classes):
    # gt_indicies1: shape = (B), 0 <= min(), max() < num_classes
    # pred_indicies2: shape = (B), 0 <= min(), max() < num_classes
    # 0 <= i < num_classes
    # returns: precision, recall, f1. shape = num_classes
    #          confusion_mat. shape = num_classes, num_classes

    confusion_mat = torch.einsum("bi, bj->ij", F.one_hot(gt_indices, num_classes=num_classes), F.one_hot(pred_indices, num_classes=num_classes))
    diag = confusion_mat[torch.arange(num_classes),torch.arange(num_classes)]
    recall = torch.nan_to_num(diag/confusion_mat.sum(dim=1), 0.0)
    precision = torch.nan_to_num(diag/confusion_mat.sum(dim=0), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
    return (precision, recall, f1), confusion_mat

def disp_cm(cm:np.ndarray, labels, fig, ax):
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 


