import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+'
'', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']

f1_df = pd.read_csv(os.path.abspath("stage_f1.csv")).rename(columns={"f1_mean":"f1"})[["f1", "f1_std"]]
f1s = f1_df["f1"]
f1_errs = f1_df["f1_std"]
plt.bar(PHASES, f1s)
plt.errorbar(PHASES, f1s, yerr=f1_errs, fmt="o", color="black")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.close()



df = pd.read_csv(os.path.abspath("wandb_stages.csv"), dtype={"Actual":str,"Predicted":str,"nPredictions":int})

confusion_mat = np.zeros((len(PHASES), len(PHASES)))
for i in range(len(PHASES)):
    for j in range(len(PHASES)):
        confusion_mat[i,j] = df[(df["Actual"] == PHASES[i]) & (df["Predicted"] == PHASES[j])].iloc[0]["nPredictions"]
confusion_mat = confusion_mat.astype(int)
fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=PHASES)
disp.plot(cmap="Blues", ax=ax, values_format='d')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.close()