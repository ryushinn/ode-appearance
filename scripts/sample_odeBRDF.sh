#!/usr/bin/env bash
# Usage:
#   scripts/sample_odeBRDF.sh <data_directory> <checkpoint_directory>
#
# Arguments:
#   $1  data_directory: The path to the directory containing the data.
#   $2  checkpoint_directory: The path to the directory where the trained ODEs was saved.

source scripts/data_SVBRDF.sh

for datum in "${data[@]}"; do
    echo "sampling with data $datum"

    python sample_odeBRDF.py --exp_path $2 --exemplars_path "$1/$datum" --comment "sampling/$datum" --checkpoint_path "$2/$datum" --renderer diffuse_cook_torrance

done
