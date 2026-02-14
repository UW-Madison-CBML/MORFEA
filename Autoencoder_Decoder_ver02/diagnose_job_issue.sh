#!/bin/bash
# diagnose_job_issue.sh

echo ""


echo ""
if [ -f "logs/extract_latents_v1_baseline.out" ]; then
    tail -20 logs/extract_latents_v1_baseline.out
else
fi

echo ""
if [ -f "logs/extract_latents_v1_baseline.err" ]; then
    cat logs/extract_latents_v1_baseline.err
else
fi

echo ""
if [ -f "logs/extract_latents_v1_baseline.log" ]; then
    tail -30 logs/extract_latents_v1_baseline.log
else
fi

echo ""
condor_submit -dry-run extract_latents_from_home.sub 2>&1 | head -20

echo ""

echo ""
condor_status -constraint 'regexp("bhaskar", Machine)' -af Name GPUs State Activity 2>/dev/null | head -10

echo ""
condor_q -hold

echo ""

