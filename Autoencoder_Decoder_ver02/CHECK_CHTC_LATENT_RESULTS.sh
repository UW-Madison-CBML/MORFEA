#!/bin/bash
# 檢查並下載 CHTC 上的 latent extraction 結果

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_BASE="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
LOCAL_DIR="./model_latents/v1_baseline"

echo "============================================================"
echo "CHTC Latent Vector Extraction 結果檢查與下載"
echo "============================================================"
echo ""

# 嘗試連接到 CHTC 並檢查結果
echo "正在連接到 CHTC 檢查結果..."
ssh -o ConnectTimeout=10 ${REMOTE_USER}@${REMOTE_HOST} << 'REMOTE_SCRIPT'
    echo "1. 檢查結果目錄："
    if [ -d /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline ]; then
        echo "   ✓ 結果目錄存在"
        ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ | head -5
    else
        echo "   ✗ 結果目錄不存在"
    fi
    
    echo ""
    echo "2. 檢查 latent 檔案數量："
    if [ -d /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents ]; then
        file_count=$(ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l)
        echo "   找到 $file_count 個 .npy 檔案"
        
        if [ $file_count -gt 0 ]; then
            echo ""
            echo "   最新的 5 個檔案："
            ls -lht /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | head -5 | awk '{print "   -", $9, "(" $5 ")"}'
            
            total_size=$(du -sh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents 2>/dev/null | cut -f1)
            echo ""
            echo "   總大小: $total_size"
        fi
    else
        echo "   ✗ latents 目錄不存在"
    fi
    
    echo ""
    echo "3. 檢查 metadata.json："
    if [ -f /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json ]; then
        echo "   ✓ metadata.json 存在"
        echo "   內容摘要："
        python3 -c "import json; d=json.load(open('/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json')); print(f\"   - 處理的胚胎數: {len(d.get('embryos', []))}\"); print(f\"   - 總序列數: {d.get('total_sequences', 'N/A')}\")" 2>/dev/null || cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json | head -10
    else
        echo "   ⚠️  metadata.json 不存在"
    fi
    
    echo ""
    echo "4. 檢查任務日誌（最後20行）："
    if [ -f ~/logs/extract_latents_v1_baseline.out ]; then
        echo "   輸出日誌："
        tail -20 ~/logs/extract_latents_v1_baseline.out
    else
        echo "   ⚠️  日誌檔案不存在"
    fi
REMOTE_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "連線成功！是否要下載結果到本地？ (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "正在下載結果..."
        mkdir -p "$LOCAL_DIR/latents"
        
        # 下載所有 .npy 檔案
        echo "下載 latent 向量檔案..."
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/latents/*.npy "$LOCAL_DIR/latents/" 2>/dev/null
        
        # 下載 metadata
        echo "下載 metadata.json..."
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/metadata.json "$LOCAL_DIR/" 2>/dev/null
        
        # 檢查下載結果
        echo ""
        echo "下載完成！本地結果："
        ls -lh "$LOCAL_DIR/latents/" 2>/dev/null | tail -5
        
        file_count=$(ls -1 "$LOCAL_DIR/latents/"*.npy 2>/dev/null | wc -l)
        echo ""
        echo "已下載 $file_count 個檔案"
    else
        echo "跳過下載"
    fi
else
    echo ""
    echo "無法連接到 CHTC，請手動執行以下命令："
    echo ""
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "ls -lh ${REMOTE_BASE}/"
fi

echo ""
echo "============================================================"

