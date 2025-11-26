"""
Cell ID Grouping Script

Groups cell_ids from a latents CSV and batches them to fit within a maximum size.
Can optionally group by embryo grades from embryo_dataset_grades.csv.
Outputs a txt file with lines containing comma-separated cell_id indices and their row ranges.

Usage:
    python get_cell_id_groups.py <latents_csv> [output_txt] [max_batch_rows] [--by-grade]

Example:
    python get_cell_id_groups.py latents.csv cell_groups.txt 10000
    python get_cell_id_groups.py latents.csv cell_groups.txt 10000 --by-grade
"""

import pandas as pd
import numpy as np
import sys
import argparse
import os


def get_cell_id_groups(csv_file, output_txt="cell_groups.txt", max_batch_rows=10000, by_grade=False, grades_file="embryo_dataset_grades.csv"):
    """
    Groups cell_ids and creates batches that fit within max_batch_rows.

    Args:
        csv_file: Path to latents CSV file
        output_txt: Output txt file to write groups
        max_batch_rows: Maximum number of rows per batch
        by_grade: If True, group by embryo grades from grades_file
        grades_file: Path to embryo grades CSV file
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

    # Load grades if grouping by grade
    if by_grade:
        if not os.path.exists(grades_file):
            print(f"\nWarning: Grades file not found: {grades_file}")
            print("Proceeding without grade grouping")
            by_grade = False
        else:
            print(f"\nLoading grades from: {grades_file}")
            grades_df = pd.read_csv(grades_file, header=None, names=['cell_id', 'grade1', 'grade2'])

            # Merge grades with cell_id_info
            cell_id_info_df = cell_id_info_df.merge(grades_df, on='cell_id', how='left')

            # Create a combined grade category
            def get_grade_category(row):
                g1 = row['grade1']
                g2 = row['grade2']

                # Handle missing grades
                if pd.isna(g1) and pd.isna(g2):
                    return 'Unknown'

                # Combine grades (e.g., "A-B", "NA-A", "B-B")
                g1_str = str(g1) if not pd.isna(g1) else 'NA'
                g2_str = str(g2) if not pd.isna(g2) else 'NA'
                return f"{g1_str}-{g2_str}"

            cell_id_info_df['grade_category'] = cell_id_info_df.apply(get_grade_category, axis=1)

            print(f"\nGrade distribution:")
            grade_counts = cell_id_info_df['grade_category'].value_counts()
            for grade, count in grade_counts.items():
                print(f"  {grade}: {count} cell_ids")

    # Batch cell_ids to fit within max_batch_rows
    if by_grade:
        # Group by grade category first, then batch within each grade
        all_batches = []
        grade_categories = cell_id_info_df['grade_category'].unique()

        for grade_cat in sorted(grade_categories):
            grade_cell_ids = cell_id_info_df[cell_id_info_df['grade_category'] == grade_cat]

            batches = []
            current_batch = []
            current_batch_rows = 0

            for _, row in grade_cell_ids.iterrows():
                cell_id = row['cell_id']
                num_rows = row['num_rows']

                # If adding this cell_id would exceed max_batch_rows, start new batch
                if current_batch and current_batch_rows + num_rows > max_batch_rows:
                    batches.append((grade_cat, current_batch))
                    current_batch = []
                    current_batch_rows = 0

                current_batch.append(cell_id)
                current_batch_rows += num_rows

            # Add final batch for this grade
            if current_batch:
                batches.append((grade_cat, current_batch))

            all_batches.extend(batches)
            print(f"\nGrade {grade_cat}: {len(batches)} batch(es)")

        batches = all_batches
    else:
        # Original batching without grade grouping
        batches = []
        current_batch = []
        current_batch_rows = 0

        for _, row in cell_id_info_df.iterrows():
            cell_id = row['cell_id']
            num_rows = row['num_rows']

            # If adding this cell_id would exceed max_batch_rows, start new batch
            if current_batch and current_batch_rows + num_rows > max_batch_rows:
                batches.append((None, current_batch))
                current_batch = []
                current_batch_rows = 0

            current_batch.append(cell_id)
            current_batch_rows += num_rows

        # Add final batch
        if current_batch:
            batches.append((None, current_batch))

    print(f"\nCreated {len(batches)} total batches (max {max_batch_rows} rows per batch)")

    # Write to output txt file
    with open(output_txt, 'w') as f:
        for batch_idx, (grade_cat, batch) in enumerate(batches):
            batch_rows = 0
            for cell_id in batch:
                cell_row = cell_id_info_df[cell_id_info_df['cell_id'] == cell_id].iloc[0]
                batch_rows += cell_row['num_rows']

            batch_line = ",".join(batch)
            f.write(batch_line + "\n")

            if grade_cat:
                print(f"  Batch {batch_idx} [{grade_cat}]: {len(batch)} cell_ids, {batch_rows} rows")
            else:
                print(f"  Batch {batch_idx}: {len(batch)} cell_ids, {batch_rows} rows")

    print(f"\nWrote cell_id groups to: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group cell_ids and batch them by size")
    parser.add_argument("csv_file", help="Path to latents CSV file")
    parser.add_argument("--output", type=str, default="cell_groups.txt", help="Output txt file")
    parser.add_argument("--max-rows", type=int, default=10000, help="Maximum rows per batch")
    parser.add_argument("--by-grade", action="store_true", help="Group by embryo grades")
    parser.add_argument("--grades-file", type=str, default="embryo_dataset_grades.csv", help="Path to grades CSV file")

    args = parser.parse_args()

    get_cell_id_groups(args.csv_file, args.output, args.max_rows, args.by_grade, args.grades_file)

