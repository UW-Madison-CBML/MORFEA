#!/bin/bash

ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
TEST_FILE="/staging/groups/bhaskar_group/ivf/test_write_permission_$(date +%s)"
if touch "$TEST_FILE" 2>/dev/null; then
    rm "$TEST_FILE"
    
    echo ""
    TEST_DIR="/staging/groups/bhaskar_group/ivf/test_dir_$(date +%s)"
    if mkdir -p "$TEST_DIR" 2>/dev/null; then
        rmdir "$TEST_DIR"
        
        echo ""
    else
        mkdir -p "$TEST_DIR" 2>&1
    fi
else
    touch "$TEST_FILE" 2>&1
fi






