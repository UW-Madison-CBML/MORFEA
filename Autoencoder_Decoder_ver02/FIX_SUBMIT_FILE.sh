#!/bin/bash
# Fix submit file - create correct version

cd ~/ivf_train

# Delete old submit file
rm -f train_h200_lab.sub

# Create correct submit file (no comments that might cause issues)
cat > train_h200_lab.sub << 'EOF'
executable = run_train.sh
arguments  =

log    = logs/train_$(Cluster)_$(Process).log
output = logs/train_$(Cluster)_$(Process).out
error  = logs/train_$(Cluster)_$(Process).err

transfer_input_files = run_train.sh, train.py, model.py, conv_lstm.py, losses.py

+WantGPULab = true
+GPUJobLength = "short"
gpus_minimum_capability = 7.0

+ProjectName = "UWMadison_BME_Bhaskar"

+SingularityImage = "docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

request_gpus   = 1
request_cpus   = 2
request_memory = 16GB
request_disk   = 30GB

Requirements = regexp("H200", TARGET.CUDADeviceName)

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_output_files   = results.tgz
transfer_output_remaps  = "results.tgz = file:///staging/groups/bhaskar_group/ivf/results/results_$(Cluster)_$(Process).tgz;"

queue 1
EOF

echo "✅ Submit file created"
echo ""
echo "File content:"
cat train_h200_lab.sub
echo ""
echo "Now submit:"
echo "  condor_submit train_h200_lab.sub"





