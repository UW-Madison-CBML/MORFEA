#!/bin/bash


SCRIPT_FILE="generate_tphate_for_aadhitya.py"
REMOTE_HOST="rho9@ap2001.chtc.wisc.edu"
REMOTE_DIR="/staging/groups/bhaskar_group/rho9"

if [ ! -f "$SCRIPT_FILE" ]; then
    exit 1
fi


scp "$SCRIPT_FILE" "$REMOTE_HOST:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo "  ssh $REMOTE_HOST"
    echo "  cd $REMOTE_DIR"
    echo "  python3 generate_tphate_for_aadhitya.py --help"
else
    exit 1
fi






