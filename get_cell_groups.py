"""
Cell ID Grouping Script

Groups cell_ids from a latents CSV and batches them to fit within a maximum size.
Outputs a txt file with lines containing comma-separated cell_id indices and their row ranges.

Usage:
    python get_cell_id_groups.py <latents_csv> [output_txt] [max_batch_rows]

Example:
    python get_cell_id_groups.py latents.csv cell_groups.txt 10000
"""

import pandas as pd
import numpy as np
import sys
import argparse


def get_cell_id_groups(csv_file, output_txt="cell_groups.txt", max_batch_rows=10000):
    """
    Groups cell_ids and creates batches that fit within max_batch_rows.

    Args:
        csv_file: Path to latents CSV file
        output_txt: Output txt file to write groups
        max_batch_rows: Maximum number of rows per batch
    """
    print(f"Loading latents from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Total samples: {len(df)}")

    # Group by cell_id and get cell_id information
    grouped_df = df.groupby("cell_id")

    cell_id_info = []
    for cell_id, group_df in grouped_df:
        start_idx = group_df.index.min()
        end_idx = group_df.index.max()
        num_rows = len(group_df)
        cell_id_info.append({
            "cell_id": cell_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_rows": num_rows
        })

    cell_id_info_df = pd.DataFrame(cell_id_info)
    print(f"\nFound {len(cell_id_info_df)} unique cell_ids")
    print(f"  Min rows per cell_id: {cell_id_info_df['num_rows'].min()}")
    print(f"  Max rows per cell_id: {cell_id_info_df['num_rows'].max()}")
    print(f"  Mean rows per cell_id: {cell_id_info_df['num_rows'].mean():.1f}")

    # Batch cell_ids to fit within max_batch_rows
    batches = []
    current_batch = []
    current_batch_rows = 0

    for _, row in cell_id_info_df.iterrows():
        cell_id = row['cell_id']
        num_rows = row['num_rows']

        # If adding this cell_id would exceed max_batch_rows, start new batch
        if current_batch and current_batch_rows + num_rows > max_batch_rows:
            batches.append(current_batch)
            current_batch = []
            current_batch_rows = 0

        current_batch.append(cell_id)
        current_batch_rows += num_rows

    # Add final batch
    if current_batch:
        batches.append(current_batch)

    print(f"\nCreated {len(batches)} batches (max {max_batch_rows} rows per batch)")

    # Write to output txt file
    with open(output_txt, 'w') as f:
        for batch_idx, batch in enumerate(batches):
            batch_rows = 0
            for cell_id in batch:
                cell_row = cell_id_info_df[cell_id_info_df['cell_id'] == cell_id].iloc[0]
                batch_rows += cell_row['num_rows']

            batch_line = ",".join(batch)
            f.write(batch_line + "\n")
            print(f"  Batch {batch_idx}: {len(batch)} cell_ids, {batch_rows} rows")

    print(f"\nWrote cell_id groups to: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group cell_ids and batch them by size")
    parser.add_argument("csv_file", help="Path to latents CSV file")
    parser.add_argument("--output", type=str, default="cell_groups.txt", help="Output txt file")
    parser.add_argument("--max-rows", type=int, default=10000, help="Maximum rows per batch")

    args = parser.parse_args()

    get_cell_id_groups(args.csv_file, args.output, args.max_rows)

