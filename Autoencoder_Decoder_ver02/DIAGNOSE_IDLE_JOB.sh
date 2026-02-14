#!/bin/bash

echo ""

condor_q -submitter rho9 -better-analyze
echo ""

condor_q -submitter rho9 -format "Cluster: %d\n" Cluster \
         -format "Process: %d\n" Process \
         -format "Status: %s\n" JobStatus \
         -format "RequestCPUs: %s\n" RequestCpus \
         -format "RequestMemory: %s\n" RequestMemory \
         -format "RequestDisk: %s\n" RequestDisk \
         -format "HoldReason: %s\n" HoldReason \
         -format "LastHoldReason: %s\n" LastHoldReason 2>/dev/null
echo ""

condor_q -submitter rho9 -hold
echo ""

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
fi

echo ""
echo ""






