#!/bin/bash
# 全面诊断所有可能的问题

echo "=== 全面诊断 ==="
echo ""

# 1. 检查任务状态
echo "1. 任务状态:"
condor_q
echo ""

# 2. 检查脚本文件
echo "2. 检查脚本文件:"
echo "   extract_latents.sh:"
if [ -f /staging/groups/bhaskar_group/rho9/extract_latents.sh ]; then
    head -3 /staging/groups/bhaskar_group/rho9/extract_latents.sh
    if grep -q "需要我提供" /staging/groups/bhaskar_group/rho9/extract_latents.sh; then
        echo "   ❌ 脚本包含占位符文本！"
    else
        echo "   ✓ 脚本没有占位符"
    fi
    echo "   脚本行数: $(wc -l < /staging/groups/bhaskar_group/rho9/extract_latents.sh)"
else
    echo "   ❌ 脚本文件不存在！"
fi
echo ""

# 3. 检查Python文件
echo "3. 检查Python文件:"
for file in extract_all_latent_trajectories.py model.py dataset_ivf.py; do
    if [ -f /staging/groups/bhaskar_group/rho9/$file ]; then
        echo "   ✓ $file 存在"
        python3 -m py_compile /staging/groups/bhaskar_group/rho9/$file 2>/dev/null && echo "     ✓ 语法正确" || echo "     ❌ 语法错误"
    else
        echo "   ❌ $file 不存在"
    fi
done
echo ""

# 4. 检查优化代码
echo "4. 检查优化代码:"
if grep -q "read directly from CSV" /staging/groups/bhaskar_group/rho9/extract_all_latent_trajectories.py; then
    echo "   ✓ 包含CSV优化"
else
    echo "   ❌ 缺少CSV优化"
fi
echo ""

# 5. 检查index.csv
echo "5. 检查index.csv:"
if [ -f /staging/groups/bhaskar_group/rho9/index.csv ]; then
    echo "   ✓ index.csv 存在"
    echo "   行数: $(wc -l < /staging/groups/bhaskar_group/rho9/index.csv)"
    if grep -q "cell_id" /staging/groups/bhaskar_group/rho9/index.csv; then
        echo "   ✓ 包含cell_id列"
    else
        echo "   ❌ 缺少cell_id列"
    fi
else
    echo "   ❌ index.csv 不存在"
fi
echo ""

# 6. 检查checkpoint
echo "6. 检查checkpoint:"
if [ -f /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt ]; then
    echo "   ✓ checkpoint存在"
    ls -lh /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt
else
    echo "   ❌ checkpoint不存在"
fi
echo ""

# 7. 检查日志文件
echo "7. 检查日志文件:"
for log in logs/extract_latents_v1_baseline.out logs/extract_latents_v1_baseline.err logs/extract_latents_v1_baseline.log; do
    if [ -f ~/$log ]; then
        echo "   $log:"
        echo "     大小: $(ls -lh ~/$log | awk '{print $5}')"
        echo "     最后修改: $(ls -l ~/$log | awk '{print $6, $7, $8}')"
    else
        echo "   ❌ $log 不存在"
    fi
done
echo ""

# 8. 检查输出日志内容
echo "8. 输出日志内容（最后20行）:"
tail -20 ~/logs/extract_latents_v1_baseline.out 2>/dev/null || echo "   日志为空或不存在"
echo ""

# 9. 检查错误日志内容
echo "9. 错误日志内容:"
cat ~/logs/extract_latents_v1_baseline.err 2>/dev/null || echo "   无错误"
echo ""

# 10. 检查condor日志中的错误
echo "10. Condor日志中的关键信息:"
tail -100 ~/logs/extract_latents_v1_baseline.log 2>/dev/null | grep -E "ERROR|error|Error|terminated|aborted" | tail -10 || echo "   无错误信息"
echo ""

# 11. 检查结果目录
echo "11. 检查结果目录:"
RESULT_DIR="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
if [ -d "$RESULT_DIR" ]; then
    echo "   ✓ 结果目录存在"
    ls -lh "$RESULT_DIR"
    if [ -d "$RESULT_DIR/latents" ]; then
        COUNT=$(ls -1 "$RESULT_DIR/latents"/*.npy 2>/dev/null | wc -l)
        echo "   Latent文件: $COUNT"
    fi
else
    echo "   ⚠️  结果目录不存在（任务可能还在处理中）"
fi
echo ""

# 12. 测试Python脚本是否可以运行
echo "12. 测试Python脚本导入:"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/staging/groups/bhaskar_group/rho9')
try:
    import extract_all_latent_trajectories
    print("   ✓ extract_all_latent_trajectories 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

try:
    import model
    print("   ✓ model 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

try:
    import dataset_ivf
    print("   ✓ dataset_ivf 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
PYEOF

echo ""
echo "=== 诊断完成 ==="

