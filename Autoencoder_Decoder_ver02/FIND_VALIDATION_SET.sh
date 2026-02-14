#!/bin/bash
# 查找或创建 validation set 列表

echo "=== 查找 Validation Set 定义 ==="
echo ""

# 检查是否有现成的 validation set 文件
echo "1. 检查是否有现成的 validation set 文件："
for file in validation_set.txt val_set.txt validation.txt val.txt; do
    if [ -f "$file" ]; then
        echo "  ✓ 找到: $file"
        echo "    包含 $(wc -l < "$file") 个 cell_id"
        head -5 "$file"
        echo ""
    fi
done

# 检查 CSV 文件是否有 split 信息
echo "2. 检查 CSV 文件是否有 split 信息："
CSV_FILE="/staging/groups/bhaskar_group/ivf/latents/latents.csv"
if [ -f "$CSV_FILE" ]; then
    echo "  检查 $CSV_FILE 的列："
    head -1 "$CSV_FILE" | tr ',' '\n' | nl
    echo ""
    
    # 检查是否有 split、set、group 等列
    if head -1 "$CSV_FILE" | grep -qiE "(split|set|group|train|val|test)"; then
        echo "  ✓ CSV 文件可能包含 split 信息"
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
echo "3. 如果需要创建 validation set，可以："
echo "   - 手动创建 validation_set.txt（每行一个 cell_id）"
echo "   - 或者告诉我 validation set 的定义方式"
echo "   - 或者先让脚本运行完所有胚胎（需要较长时间）"






