#!/bin/bash

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_BASE="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
LOCAL_DIR="./model_latents/v1_baseline"

echo "============================================================"
echo "============================================================"
echo ""

ssh -o ConnectTimeout=10 ${REMOTE_USER}@${REMOTE_HOST} << 'REMOTE_SCRIPT'
    if [ -d /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline ]; then
        ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ | head -5
    else
    fi
    
    echo ""
    if [ -d /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents ]; then
        file_count=$(ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l)
        
        if [ $file_count -gt 0 ]; then
            echo ""
            ls -lht /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | head -5 | awk '{print "   -", $9, "(" $5 ")"}'
            
            total_size=$(du -sh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents 2>/dev/null | cut -f1)
            echo ""
        fi
    else
    fi
    
    echo ""
    if [ -f /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json ]; then
        python3 -c "import json; d=json.load(open('/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json')); print(f\"   - 處理的胚胎數: {len(d.get('embryos', []))}\"); print(f\"   - 總序列數: {d.get('total_sequences', 'N/A')}\")" 2>/dev/null || cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json | head -10
    else
    fi
    
    echo ""
    if [ -f ~/logs/extract_latents_v1_baseline.out ]; then
        tail -20 ~/logs/extract_latents_v1_baseline.out
    else
    fi
REMOTE_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        mkdir -p "$LOCAL_DIR/latents"
        
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/latents/*.npy "$LOCAL_DIR/latents/" 2>/dev/null
        
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/metadata.json "$LOCAL_DIR/" 2>/dev/null
        
        echo ""
        ls -lh "$LOCAL_DIR/latents/" 2>/dev/null | tail -5
        
        file_count=$(ls -1 "$LOCAL_DIR/latents/"*.npy 2>/dev/null | wc -l)
        echo ""
    else
    fi
else
    echo ""
    echo ""
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "ls -lh ${REMOTE_BASE}/"
fi

echo ""
echo "============================================================"

