#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo "   condor_q -submitter rho9 | grep extract"
echo ""
echo "   cat ~/logs/extract_latents_v1_baseline.out | grep -A 5 -B 5 'Device:'"
echo ""
echo "   condor_q -submitter rho9"
echo "   condor_history -limit 1 -submitter rho9 | grep extract"
echo ""
echo "   condor_history -limit 1 2853038 -long | grep -E 'ExitCode|ExitStatus'"
echo ""
echo "============================================================"
echo "============================================================"








