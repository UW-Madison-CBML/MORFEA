#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo "   condor_q -long 2852953.0 | grep -E 'JobStatus|RemoteHost|RemoteStartTime|JobCurrentStartDate'"
echo ""
echo "   condor_tail 2852953.0 | tail -30"
echo ""
echo "   grep -E 'Job executing|Job terminated|Job was evicted|Job was held' ~/logs/extract_latents_v1_baseline.log | tail -10"
echo ""
echo "   condor_tail -f 2852953.0"
echo ""
echo "============================================================"
echo ""
echo "============================================================"

