import pandas as pd
import numpy as np
import os
df = pd.read_csv('latents.csv')
grouped_df = df.groupby("cell_id")
size_df = pd.DataFrame({"cell_id": [cell_id for (cell_id, _) in grouped_df], "length": [group_df.shape[0] for (_, group_df) in grouped_df]})

