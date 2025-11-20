# generate_test_latents.py - Generate random latent embeddings CSV for testing

import numpy as np
import pandas as pd
import argparse


def generate_test_latents_csv(
    num_cells: int = 5,
    time_steps_per_cell: int = 50,
    latent_dims: int = 200,
    output_csv: str = "test_latents.csv",
    random_seed: int = 42
):
    """
    Generate random latent embeddings for testing visualize.py

    Args:
        num_cells: Number of unique cell IDs
        time_steps_per_cell: Time steps per cell sequence
        latent_dims: Number of latent dimensions (default 200)
        output_csv: Output CSV filename
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)

    # Generate data
    cell_ids = []
    time_steps = []
    latent_data = []

    print(f"Generating test data:")
    print(f"  Cells: {num_cells}")
    print(f"  Time steps per cell: {time_steps_per_cell}")
    print(f"  Latent dimensions: {latent_dims}")

    for cell_idx in range(num_cells):
        cell_id = f"cell_{cell_idx:03d}"
        for t in range(time_steps_per_cell):
            # Generate random latent vector (mean=0, std=1)
            z = np.random.randn(latent_dims)

            latent_data.append(z)
            cell_ids.append(cell_id)
            time_steps.append(t)

    # Create DataFrame matching export_latents.py format
    latent_columns = [f"z_{i}" for i in range(latent_dims)]
    latents_array = np.array(latent_data)

    df = pd.DataFrame(latents_array, columns=latent_columns)
    df.insert(0, "cell_id", cell_ids)
    df.insert(1, "time_step", time_steps)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"\nGenerated CSV: {output_csv}")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random test latents CSV")
    parser.add_argument("--cells", type=int, default=5, help="Number of cells")
    parser.add_argument("--timesteps", type=int, default=50, help="Time steps per cell")
    parser.add_argument("--dims", type=int, default=200, help="Latent dimensions")
    parser.add_argument("--output", type=str, default="test_latents.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_test_latents_csv(
        num_cells=args.cells,
        time_steps_per_cell=args.timesteps,
        latent_dims=args.dims,
        output_csv=args.output,
        random_seed=args.seed
    )
