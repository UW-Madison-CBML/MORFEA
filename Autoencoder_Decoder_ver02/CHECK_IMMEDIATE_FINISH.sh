#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo "   condor_history -limit 1 2853038 -long | grep -E 'ExitCode|ExitStatus|HoldReason'"
echo ""
echo "   cat ~/logs/extract_latents_v1_baseline.err"
echo ""
echo "   cat ~/logs/extract_latents_v1_baseline.out"
echo ""
echo "   tail -50 /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out 2>/dev/null"
echo "   cat /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.err 2>/dev/null"
echo ""
echo "   grep '2853038' ~/logs/extract_latents_v1_baseline.log | tail -20"
echo ""
echo "============================================================"








