#!/bin/bash

# Bash script to run attr_patch.py with multiple head indices

# Array of head indices to process
HEADS=(29 30 31)

# Loop through each head index
for head in "${HEADS[@]}"; do
    echo "========================================"
    echo "Processing head $head"
    echo "========================================"
    python attr_patch.py --head $head
    echo ""
done

echo "All head analyses complete!"
