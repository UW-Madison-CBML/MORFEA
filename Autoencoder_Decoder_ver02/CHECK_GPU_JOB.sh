#!/bin/bash

condor_q -submitter rho9 -better-analyze
echo ""

condor_status -constraint 'GPUs > 0' -format "Name: %s\n" Name -format "GPUs: %d\n" GPUs -format "State: %s\n" State | head -20
echo ""

echo ""







