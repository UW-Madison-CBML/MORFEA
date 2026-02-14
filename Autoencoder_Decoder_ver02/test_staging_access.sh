#!/bin/bash
# 測試腳本：在提交任務前驗證 staging 訪問

set -e

echo "=== 測試 staging 訪問配置 ==="
echo ""

# 1. 檢查 train_ivf.sub 配置
echo "1. 檢查 train_ivf.sub 配置..."
if [ ! -f "train_ivf.sub" ]; then
    echo "   ✗ ERROR: train_ivf.sub 不存在"
    exit 1
fi

echo "   ✓ train_ivf.sub 存在"

# 檢查 SingularityBind
if grep -q "SingularityBind.*staging" train_ivf.sub; then
    BIND_LINE=$(grep "SingularityBind" train_ivf.sub)
    echo "   ✓ 找到 SingularityBind: $BIND_LINE"
    
    # 檢查格式是否正確
    if echo "$BIND_LINE" | grep -q "/staging:/staging"; then
        echo "   ✓ SingularityBind 格式正確"
    else
        echo "   ⚠ WARNING: SingularityBind 格式可能不正確"
        echo "     建議使用: +SingularityBind = \"/staging:/staging\""
    fi
else
    echo "   ✗ ERROR: train_ivf.sub 中未找到 SingularityBind"
    exit 1
fi

# 檢查 Requirements
if grep -q "HasCHTCStaging.*true" train_ivf.sub; then
    echo "   ✓ 已設置 HasCHTCStaging == true"
else
    echo "   ⚠ WARNING: 未明確設置 HasCHTCStaging"
fi

echo ""

# 2. 檢查 run_train.sh 邏輯
echo "2. 檢查 run_train.sh 邏輯..."
if [ ! -f "run_train.sh" ]; then
    echo "   ✗ ERROR: run_train.sh 不存在"
    exit 1
fi

echo "   ✓ run_train.sh 存在"

# 檢查是否嘗試訪問 staging
if grep -q "/staging/groups/bhaskar_group" run_train.sh; then
    echo "   ✓ run_train.sh 包含 staging 路徑"
    
    # 檢查個人目錄路徑
    if grep -q "/staging/groups/bhaskar_group/rho9/ivf_data" run_train.sh; then
        echo "   ✓ 包含個人 staging 目錄路徑"
    else
        echo "   ⚠ WARNING: 未包含個人 staging 目錄路徑"
    fi
else
    echo "   ✗ ERROR: run_train.sh 中未找到 staging 路徑"
    exit 1
fi

# 檢查是否使用 head -c 1 測試權限
if grep -q "head -c 1" run_train.sh; then
    echo "   ✓ 使用 head -c 1 測試文件訪問權限"
else
    echo "   ⚠ WARNING: 未使用 head -c 1 測試權限"
fi

# 檢查是否複製文件
if grep -q "cp.*embryo_dataset.tar.gz" run_train.sh; then
    echo "   ✓ 包含複製文件到工作目錄的邏輯"
else
    echo "   ⚠ WARNING: 未找到複製文件到工作目錄的邏輯"
fi

echo ""

# 3. 在 submit 節點測試 staging 訪問（如果可能）
echo "3. 在 submit 節點測試 staging 訪問..."
POSSIBLE_TARS=(
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
)

FOUND_ON_SUBMIT=""
for tar_path in "${POSSIBLE_TARS[@]}"; do
    if [ -r "$tar_path" ]; then
        FOUND_ON_SUBMIT="$tar_path"
        echo "   ✓ 在 submit 節點可以訪問: $tar_path"
        echo "      文件大小: $(ls -lh "$tar_path" | awk '{print $5}')"
        break
    else
        echo "   - 在 submit 節點無法訪問: $tar_path"
    fi
done

if [ -z "$FOUND_ON_SUBMIT" ]; then
    echo "   ✗ ERROR: 在 submit 節點無法訪問任何 staging 路徑"
    echo "      請確認文件路徑和權限"
    exit 1
fi

echo ""

# 4. 檢查必要文件
echo "4. 檢查必要文件..."
REQUIRED_FILES=(
    "train.py"
    "dataset_ivf.py"
    "losses.py"
    "model.py"
    "conv_lstm.py"
    "build_index.py"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file 存在"
    else
        echo "   ✗ $file 不存在"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" != "true" ]; then
    echo "   ✗ ERROR: 缺少必要文件"
    exit 1
fi

echo ""

# 5. 總結
echo "=== 配置檢查總結 ==="
echo "✓ train_ivf.sub 配置正確"
echo "✓ run_train.sh 邏輯正確"
echo "✓ submit 節點可以訪問 staging"
echo "✓ 所有必要文件存在"
echo ""
echo "✅ 配置檢查通過！可以提交任務。"
echo ""
echo "提交前請確認："
echo "1. 已在 CHTC 上傳所有文件"
echo "2. 已刪除舊的失敗任務 (condor_rm <job_id>)"
echo "3. 準備好監控任務狀態 (condor_q, condor_tail)"
echo ""

