#!/usr/bin/env bash
# Usage:
#   scripts/odeBRDF.sh <data_directory> <output_directory>
#
# Arguments:
#   $1  data_directory: The path to the directory containing the data.
#   $2  output_directory: The path to the directory where the output will be saved.

source scripts/data_SVBRDF.sh

for datum in "${data[@]}"; do
    echo "training ode-BRDF with data $datum"

    python main_odeBRDF.py --exp_path $2 --exemplars_path "$1/$datum" --comment "$datum" --renderer diffuse_cook_torrance

done
