#!/usr/bin/env python3
"""
List all available cell IDs in the dataset
"""

from pathlib import Path
import sys

def list_cells(data_root="data"):
    """List all available cell directories"""
    data_path = Path(data_root)
    
    # Try alternative paths
    if not data_path.exists():
        if Path('data').exists():
            data_path = Path('data')
        elif Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset').exists():
            data_path = Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
        else:
            print(f"Error: Data root not found: {data_root}")
            return []
    
    cells = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    return cells

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='List available cell IDs')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Data root directory (default: auto-detect)')
    args = parser.parse_args()
    
    if args.data_root:
        data_root = args.data_root
    else:
        # Auto-detect
        if Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset').exists():
            data_root = '/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset'
        elif Path('data').exists():
            data_root = 'data'
        else:
            data_root = 'data'
    
    cells = list_cells(data_root)
    
    print(f"Found {len(cells)} cell directories in {data_root}")
    print("\nAvailable cells:")
    for i, cell in enumerate(cells, 1):
        print(f"  {i:3d}. {cell}")
    
    # Check for specific cells mentioned in requirements
    print("\nTarget cells (from requirements):")
    target_cells = ['ZS435-5', 'RS363-7']
    for target in target_cells:
        if target in cells:
            print(f"  ✓ {target} - Found")
        else:
            print(f"  ✗ {target} - Not found")

