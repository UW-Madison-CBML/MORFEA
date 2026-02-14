#!/bin/bash

echo ""

ping -c 3 ap2001.chtc.wisc.edu 2>&1 | head -5

echo ""

echo ""
nslookup ap2001.chtc.wisc.edu 2>&1 | head -5

echo ""
echo ""








