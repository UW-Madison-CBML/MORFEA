#!/bin/bash

echo ""

POSSIBLE_PATHS=(
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/ivf/data"
    "/staging/groups/bhaskar_group/ivf/latents"
    "/staging/groups/bhaskar_group/rho9/ivf"
    "/staging/groups/bhaskar_group/rho9/ivf/data"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -f "$path/latents.npy" ]; then
            ls -lh "$path/latents.npy"
        fi
        if [ -f "$path/latents.csv" ]; then
            ls -lh "$path/latents.csv"
        fi
        ls -lh "$path" | head -10
        echo ""
    fi
done

echo ""
find /staging/groups/bhaskar_group -name "latents.npy" -type f 2>/dev/null | head -10

echo ""
find /staging/groups/bhaskar_group -name "latents.csv" -type f 2>/dev/null | head -10

echo ""
find /staging/groups/bhaskar_group -type d -name "*ivf*" 2>/dev/null

echo ""






