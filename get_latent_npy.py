import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('latents.csv')

# Extract the latent columns (lat_0 through lat_4095)
lat_columns = [f'z_{i}' for i in range(4096)]
latent_data = df[lat_columns].values  # Shape: (num_rows, 4096)

# Save the latent data as npy
np.save('latents.npy', latent_data)

# Save cell_id and timestep as CSV
metadata = df[['cell_id', 'time_step']]
metadata.to_csv('metadata.csv', index=False)

print(f"Saved {latent_data.shape[0]} rows with shape {latent_data.shape}")
print(f"npy file: latent_data.npy")
print(f"csv file: metadata.csv")
