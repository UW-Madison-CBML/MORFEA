#!/bin/bash

echo ""

cd ~

cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).log
output = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).out
error = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).err
getenv = True
initialdir = /staging/groups/bhaskar_group/rho9
executable = /usr/bin/python3
arguments = generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = NO
queue 1
EOF

echo "✓ Created generate_tphate.sub in home directory"
echo ""

mkdir -p /staging/groups/bhaskar_group/rho9/logs

echo "Submitting job..."
condor_submit generate_tphate.sub

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Monitor with:"
    echo "  condor_q -submitter rho9"
    echo "  condor_tail -f <ClusterID>"
else
    echo ""
    echo "❌ Submission failed"
fi






