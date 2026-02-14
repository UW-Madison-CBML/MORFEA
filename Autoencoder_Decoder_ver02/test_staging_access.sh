#!/bin/bash

set -e

echo ""

if [ ! -f "train_ivf.sub" ]; then
    exit 1
fi


if grep -q "SingularityBind.*staging" train_ivf.sub; then
    BIND_LINE=$(grep "SingularityBind" train_ivf.sub)
    
    if echo "$BIND_LINE" | grep -q "/staging:/staging"; then
    else
    fi
else
    exit 1
fi

if grep -q "HasCHTCStaging.*true" train_ivf.sub; then
else
fi

echo ""

if [ ! -f "run_train.sh" ]; then
    exit 1
fi


if grep -q "/staging/groups/bhaskar_group" run_train.sh; then
    
    if grep -q "/staging/groups/bhaskar_group/rho9/ivf_data" run_train.sh; then
    else
    fi
else
    exit 1
fi

if grep -q "head -c 1" run_train.sh; then
else
fi

if grep -q "cp.*embryo_dataset.tar.gz" run_train.sh; then
else
fi

echo ""

POSSIBLE_TARS=(
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
)

FOUND_ON_SUBMIT=""
for tar_path in "${POSSIBLE_TARS[@]}"; do
    if [ -r "$tar_path" ]; then
        FOUND_ON_SUBMIT="$tar_path"
        break
    else
    fi
done

if [ -z "$FOUND_ON_SUBMIT" ]; then
    exit 1
fi

echo ""

REQUIRED_FILES=(
    "train.py"
    "dataset_ivf.py"
    "losses.py"
    "model.py"
    "conv_lstm.py"
    "build_index.py"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
    else
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" != "true" ]; then
    exit 1
fi

echo ""

echo ""
echo ""
echo ""

