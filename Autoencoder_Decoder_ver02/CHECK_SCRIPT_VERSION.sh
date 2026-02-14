#!/bin/bash
# 检查脚本是否支持 --start_from 参数

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查脚本版本 ==="
if grep -q "start_from" generate_tphate_for_aadhitya.py; then
    echo "✓ 脚本支持 --start_from 参数"
else
    echo "⚠️  脚本不支持 --start_from 参数（需要更新）"
    echo ""
    echo "需要从本地上传更新后的脚本："
    echo "  cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
    echo "  scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
fi






