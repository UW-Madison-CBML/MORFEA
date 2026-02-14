#!/bin/bash

echo ""

for file in validation_set.txt val_set.txt validation.txt val.txt; do
    if [ -f "$file" ]; then
        head -5 "$file"
        echo ""
    fi
done

CSV_FILE="/staging/groups/bhaskar_group/ivf/latents/latents.csv"
if [ -f "$CSV_FILE" ]; then
    head -1 "$CSV_FILE" | tr ',' '\n' | nl
    echo ""
    
    if head -1 "$CSV_FILE" | grep -qiE "(split|set|group|train|val|test)"; then
        python3 << 'EOF'
import pandas as pd
df = pd.read_csv('/staging/groups/bhaskar_group/ivf/latents/latents.csv')
print(f"Columns: {df.columns.tolist()}")
if 'split' in df.columns or 'set' in df.columns:
    print(f"\nUnique values in split/set column:")
    col = 'split' if 'split' in df.columns else 'set'
    print(df[col].value_counts())
EOF
    fi
fi

echo ""






