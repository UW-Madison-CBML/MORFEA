import torch
from signature_dataset import SignatureDataset
from signature_model import SignatureClassifier
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        """Add a new value and update statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        """Returns sample variance (unbiased). Use self.m2 / self.n for population."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_dev(self):
        """Returns sample standard deviation."""
        return math.sqrt(self.variance)

def evaluate_model_detailed(model, loader, criterion, device,keep_na=False, grade_name="Grade"):
    """Comprehensive evaluation with per-class metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for sig, labels in loader:
            sig = sig.to(device)
            labels = labels.to(device).long()
            
            logits = model(sig)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall metrics
    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples
    overall_acc = (all_preds == all_labels).mean()
    
    # Per-class metrics
    grade_names = ["A","B","C","NA"] if keep_na else ["A", "B", "C"]
    class_stats = {}
    
    for cls in range(4 if keep_na else 3):
        mask = all_labels == cls
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_count = mask.sum()
            class_stats[grade_names[cls]] = {
                'accuracy': class_acc,
                'count': class_count,
                'percentage': class_count / total_samples
            }
        else:
            class_stats[grade_names[cls]] = {
                'accuracy': 0.0,
                'count': 0,
                'percentage': 0.0
            }
    
    # Confusion matrix
    confusion = np.zeros((4,4) if keep_na else (3, 3), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    
    # Random baseline (weighted)
    class_proportions = np.array([class_stats[g]['percentage'] for g in grade_names])
    random_baseline = (class_proportions ** 2).sum()
    
    # Majority baseline
    majority_baseline = class_proportions.max()
    
    # Improvement metrics
    improvement_over_random = (overall_acc - random_baseline) / (1 - random_baseline) * 100
    improvement_over_majority = (overall_acc - majority_baseline) / (1 - majority_baseline) * 100
    
    return {
        'loss': avg_loss,
        'accuracy': overall_acc,
        'class_stats': class_stats,
        'confusion': confusion,
        'random_baseline': random_baseline,
        'majority_baseline': majority_baseline,
        'improvement_over_random': improvement_over_random,
        'improvement_over_majority': improvement_over_majority,
        'total_samples': total_samples
    }


def print_evaluation_report(results, grade_type, keep_na=False):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{grade_type} MODEL EVALUATION")
    print(f"{'='*60}")
    
    print(f"\nOverall Performance:")
    print(f"  Loss:     {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.1%} ({int(results['accuracy'] * results['total_samples'])}/{results['total_samples']})")
    
    print(f"\nBaselines:")
    print(f"  Random guessing:     {results['random_baseline']:.1%}")
    print(f"  Majority class:      {results['majority_baseline']:.1%}")
    
    print(f"\nImprovement:")
    print(f"  vs Random:   {results['improvement_over_random']:.1f}% of possible improvement")
    print(f"  vs Majority: {results['improvement_over_majority']:.1f}% of possible improvement")
    
    print(f"\nPer-Class Performance:")
    for grade in ["A","B","C", "NA"] if keep_na else ["A", "B", "C"]:
        stats = results['class_stats'][grade]
        print(f"  Grade {grade}: {stats['accuracy']:.1%} "
              f"({stats['count']:3d} samples, {stats['percentage']:.1%} of dataset)")
    
    print(f"\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"             A    B    C")
    print(f"  Actual A  {results['confusion'][0,0]:3d}  {results['confusion'][0,1]:3d}  {results['confusion'][0,2]:3d}")
    print(f"         B  {results['confusion'][1,0]:3d}  {results['confusion'][1,1]:3d}  {results['confusion'][1,2]:3d}")
    print(f"         C  {results['confusion'][2,0]:3d}  {results['confusion'][2,1]:3d}  {results['confusion'][2,2]:3d}")
    
    # Identify common mistakes
    conf = results['confusion']
    print(f"\nCommon Misclassifications:")
    for true_cls in range(3):
        for pred_cls in range(3):
            if true_cls != pred_cls and conf[true_cls, pred_cls] > 0:
                grade_names = ["A", "B", "C"]
                pct = conf[true_cls, pred_cls] / conf[true_cls].sum() * 100
                print(f"  {grade_names[true_cls]} → {grade_names[pred_cls]}: "
                      f"{conf[true_cls, pred_cls]} times ({pct:.1f}%)")

VAL_EMBRYOS =[
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
    ]
def main(model_name):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
    )
    
    learning_rate = 0.001
    sigs_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv")).rename(columns={"embryo_id":"cell_id"})
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=False)
    mask = sigs_df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = sigs_df[mask]
    print(len(val_df)/len(sigs_df))
    sigs_df = sigs_df[~mask]


    dataset_te = SignatureDataset(sigs_df, grades_df, "TE", keep_na=True) 
    dataset_icm = SignatureDataset(sigs_df, grades_df, "ICM", keep_na=True)
    dataset_te_val = SignatureDataset(val_df, grades_df, "TE", keep_na=True) 
    dataset_icm_val = SignatureDataset(val_df, grades_df, "ICM", keep_na=True)
    sig_size = len([i for i in sigs_df.columns if i[:2] == "s_"])
    crit_te = torch.nn.CrossEntropyLoss()
    crit_icm = torch.nn.CrossEntropyLoss()
    model_te = SignatureClassifier(sig_size, keep_na=True)
    model_te = model_te.to(DEVICE)
    model_icm = SignatureClassifier(sig_size, keep_na=True)
    model_icm = model_icm.to(DEVICE)
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=learning_rate, weight_decay=1e-5)


    loader_te = DataLoader(
        dataset_te,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    print(f"TE train dataset size: {len(dataset_te)}")
    print(f"TE val dataset size: {len(dataset_te_val)}")
    print(f"ICM train dataset size: {len(dataset_icm)}")
    print(f"ICM val dataset size: {len(dataset_icm_val)}")

    print(f"TE val loader batches: {len(loader_te_val)}")
    print(f"ICM val loader batches: {len(loader_icm_val)}")

    for epoch in range(200):
        model_te.train(); model_icm.train()
        for sig, te in loader_te:
            sig = sig.to(DEVICE)
            te = te.to(DEVICE).long()
            label = model_te(sig)
            loss = crit_te(label, te)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()
            run.log({"te": loss.item()})

        for sig, icm in loader_icm:
            sig = sig.to(DEVICE)
            icm = icm.to(DEVICE).long()

            label = model_icm(sig)
            loss = crit_icm(label, icm)

            optimizer_icm.zero_grad() 
            loss.backward() 
            optimizer_icm.step()
            run.log({"icm": loss.item()})
    
    te_loss_stats = RunningStats()
    icm_loss_stats = RunningStats()
    te_acc_stats = RunningStats()
    icm_acc_stats = RunningStats()

    model_te.eval(); model_icm.eval()
    with torch.no_grad():
        for sig, te in loader_te_val:
            sig = sig.to(DEVICE)
            te = te.to(DEVICE).long()
            logits = model_te(sig)
            loss = crit_te(logits, te)
            te_loss_stats.push(loss.item())
            
            # Calculate accuracy
            preds = logits.argmax(dim=1)  # Get predicted class (0, 1, or 2)
            te_acc_stats.push((preds == te).sum().item()/te.shape[0])
        for sig, icm in loader_icm_val:
            sig = sig.to(DEVICE)
            icm = icm.to(DEVICE).long()

            logits = model_icm(sig)
            loss = crit_icm(logits, icm)
            icm_loss_stats.push(loss.item())
        
            # Calculate accuracy
            preds = logits.argmax(dim=1)  # Get predicted class (0, 1, or 2)
            icm_acc_stats.push((preds == icm).sum().item()/icm.shape[0])

    print("TE: " + str(te_loss_stats.mean) + " +- " + str(te_loss_stats.std_dev))
    print("ICM: " + str(icm_loss_stats.mean) + " +- " + str(icm_loss_stats.std_dev))
    print("TE Acc: " + str(te_acc_stats.mean) + " +- " + str(te_acc_stats.std_dev))
    print("ICM Acc: " + str(icm_acc_stats.mean) + " +- " + str(icm_acc_stats.std_dev))

    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    # Evaluate TE model
    te_results = evaluate_model_detailed(model_te, loader_te_val, crit_te, DEVICE, "TE", keep_na=True)
    
    # Evaluate ICM model
    icm_results = evaluate_model_detailed(model_icm, loader_icm_val, crit_icm, DEVICE, "ICM", keep_na=True)
    
    # Log to wandb
    run.log({
        "final_te_accuracy": te_results['accuracy'],
        "final_te_random_baseline": te_results['random_baseline'],
        "final_te_majority_baseline": te_results['majority_baseline'],
        "final_te_improvement_random": te_results['improvement_over_random'],
        "final_te_improvement_majority": te_results['improvement_over_majority'],
        "final_icm_accuracy": icm_results['accuracy'],
        "final_icm_random_baseline": icm_results['random_baseline'],
        "final_icm_majority_baseline": icm_results['majority_baseline'],
        "final_icm_improvement_random": icm_results['improvement_over_random'],
        "final_icm_improvement_majority": icm_results['improvement_over_majority'],
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)
