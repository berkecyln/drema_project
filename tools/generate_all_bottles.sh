#!/bin/bash
set -e

PROJECT=/home/gunnleif/Projects/drema/drema_project
cd $PROJECT

BOTTLES=(
    "gello_bottle1_rawtsdf"
    "gello_bottle2_rawtsdf"
    "gello_bottle3_rawtsdf"
    "gello_bottle4_rawtsdf"
)

for bottle in "${BOTTLES[@]}"; do
    echo "========================================"
    echo "Generating: $bottle"
    echo "========================================"
    python generate_new_data.py data.source_path=$PROJECT/input/$bottle
    echo "Done: $bottle"
done

echo "All bottles done."
