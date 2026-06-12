import torch
import torch.nn.functional as F
def prfcm(gt_indices, pred_indices, num_classes):
    # gt_indicies1: shape = (B), 0 <= min(), max() < num_classes
    # pred_indicies2: shape = (B), 0 <= min(), max() < num_classes
    # 0 <= i < num_classes
    # returns: precision, recall, f1. shape = num_classes
    #          confusion_mat. shape = num_classes, num_classes

    confusion_mat = torch.einsum("bi, bj->ij", F.one_hot(pred_indices, num_classes=num_classes), F.one_hot(gt_indices, num_classes=num_classes))
    diag = confusion_mat[torch.arange(num_classes),torch.arange(num_classes)]
    recall = torch.nan_to_num(diag/confusion_mat.sum(dim=0), 0.0)
    precision = torch.nan_to_num(diag/confusion_mat.sum(dim=1), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
    return recall, precision, f1, confusion_mat


