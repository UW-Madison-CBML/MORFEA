#!/bin/bash
# 在 CHTC 上執行此腳本，修復 TPHATE 調用

cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

# 備份
cp analyze_trajectory_curvature.py analyze_trajectory_curvature.py.bak3

python3 << 'PYEOF'
with open('analyze_trajectory_curvature.py', 'r') as f:
    content = f.read()

# 找到並替換 compute_tphate_3d 函數中的 TPHATE 調用部分
import re

# 替換 1: 移除 time_edges 參數
content = re.sub(
    r'embedding = tph\.fit_transform\(latents, time_edges=time_edges\)',
    'embedding = tph.fit_transform(latents)',
    content
)

# 替換 2: 移除所有 PHATE fallback 代碼，只保留 TPHATE
# 找到 if TPHATE_AVAILABLE 區塊並替換
old_pattern = r'if TPHATE_AVAILABLE:.*?raise RuntimeError\("Neither TPHATE nor PHATE available"\)'
new_code = '''if not TPHATE_AVAILABLE:
        raise RuntimeError("TPHATE is REQUIRED but not available. Please install tphate.")
    
    print("Calculating TPHATE...")
    try:
        # Create TPHATE instance
        # TPHATE automatically infers temporal structure from data order
        tph = tphate.TPHATE(n_components=n_components, knn=knn, verbose=1)
        
        # Fit and transform (TPHATE infers temporal structure from data order)
        embedding = tph.fit_transform(latents)
        
        print(f"  ✓ TPHATE embedding shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        error_msg = f"TPHATE failed: {e}\\nTPHATE is REQUIRED. No PHATE fallback allowed.\\nPlease check tphate installation and try again."
        raise RuntimeError(error_msg)'''

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# 刪除所有 else 分支中的 PHATE 代碼
content = re.sub(
    r'else:.*?raise RuntimeError\("TPHATE not available.*?"\)',
    '',
    content,
    flags=re.DOTALL
)

# 刪除重複的 else 和 PHATE_AVAILABLE 檢查
content = re.sub(
    r'else:\s+if PHATE_AVAILABLE:.*?raise RuntimeError\("Neither TPHATE nor PHATE available"\)',
    '',
    content,
    flags=re.DOTALL
)

with open('analyze_trajectory_curvature.py', 'w') as f:
    f.write(content)

print("✓ 已修復：移除 time_edges，移除 PHATE fallback")

# 驗證語法
import py_compile
try:
    py_compile.compile('analyze_trajectory_curvature.py', doraise=True)
    print("✓ 語法檢查通過")
except py_compile.PyCompileError as e:
    print(f"❌ 語法錯誤: {e}")
    import shutil
    shutil.copy('analyze_trajectory_curvature.py.bak3', 'analyze_trajectory_curvature.py')
    print("已恢復備份")
    exit(1)
PYEOF

echo ""
echo "✓ 修復完成！現在可以測試："
echo "cd /staging/groups/bhaskar_group/rho9/ivf_analysis"
echo "python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5"

