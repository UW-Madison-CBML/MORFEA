#!/bin/bash

ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
id

echo ""
ls -la /staging/groups/bhaskar_group/ivf/ | head -20

echo ""
ls -ld /staging/groups/bhaskar_group/ivf/latents/

echo ""
TEST_FILE="/staging/groups/bhaskar_group/ivf/latents/test_$(date +%s)"
touch "$TEST_FILE" 2>&1
if [ $? -eq 0 ]; then
    rm "$TEST_FILE"
else
    touch "$TEST_FILE" 2>&1
fi






