import torch
from signature_dataset import SignatureDataset
from signature_model import SignatureModel
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset

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

def main(model_name):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
    )
    
    learning_rate = 2e-4
    optimizer_te = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_icm = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    sigs_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv"))
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"))
    dataset = SignatureDataset(sigs_df, grades_df) 
    sig_size = len([i for i in sigs_df.columns if i[:2] == "s_"])
    crit_te = torch.nn.CrossEntropyLoss()
    crit_icm = torch.nn.CrossEntropyLoss()
    model_te = SignatureModel(sig_size)
    model_te = model_te.to(DEVICE)
    model_icm = SignatureModel(sig_size)
    model_icm = model_icm.to(DEVICE)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    for epoch in range(10):
        model.train()
        for sig, te, icm in loader:
            sig = sig.squeeze().to(DEVICE)
            te = te.squeeze()[:3].to(DEVICE)
            icm = icm.squeeze()[:3].to(DEVICE)
            if(not torch.all(torch.eq(te, 0))):
                label = model_te(sig)
                loss = crit_te(label, te)

                optimizer_te.zero_grad() 
                loss.backward() 
                optimizer_te.step()
                run.log({"te": loss.item()})
            if(not torch.all(torch.eq(icm, 0))):
                label = model_te(sig)
                loss = crit_icm(label, icm)

                optimizer_icm.zero_grad() 
                loss.backward() 
                optimizer_icm.step()
                run.log({"icm": loss.item()})
    
    te_stats = RunningStats()
    icm_stats = RunningStats()
    model.eval()
    with torch.no_grad():
        for sig, te, icm in loader:
            sig = sig.squeeze().to(DEVICE)
            te = te.squeeze()[:3].to(DEVICE)
            icm = icm.squeeze()[:3].to(DEVICE)
            if(not torch.all(torch.eq(te, 0))):
                label = model_te(sig)
                loss = crit_te(label, te)

                te_stats.push(loss.item())
            if(not torch.all(torch.eq(icm, 0))):
                label = model_te(sig)
                loss = crit_icm(label, icm)

                icm_stats.push(loss.item())
    print("TE: " + str(te_stats.mean) + " +- " + str(te_stats.std_dev))
    print("ICM: " + str(icm_stats.mean) + " +- " + str(icm_stats.std_dev))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)
