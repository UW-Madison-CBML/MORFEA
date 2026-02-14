#!/bin/bash
# 在 CHTC 上创建 submit 文件（单行版本，避免语法错误）

cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = logs/generate_tphate_$(Cluster).log
output = logs/generate_tphate_$(Cluster).out
error = logs/generate_tphate_$(Cluster).err
getenv = True

executable = /usr/bin/python3
arguments = generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base aadhitya_v1_val --knn 5 --skip_existing

request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = aadhitya_v1_val

queue 1
EOF

echo "✓ Created generate_tphate.sub"
echo ""
echo "Now you can:"
echo "  1. mkdir -p logs"
echo "  2. condor_submit generate_tphate.sub"






