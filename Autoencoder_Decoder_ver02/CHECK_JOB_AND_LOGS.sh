#!/bin/bash

condor_q -submitter rho9
echo ""

find ~/logs -name "generate_tphate_*" -type f 2>/dev/null | head -5
echo ""

echo ""

find ~ -name "*generate_tphate*" -type f 2>/dev/null | head -10
echo ""

condor_q -submitter rho9 -format "Cluster: %d\n" Cluster -format "Process: %d\n" Process -format "Status: %s\n" JobStatus -format "Log: %s\n" UserLog 2>/dev/null | head -20
echo ""

echo ""






