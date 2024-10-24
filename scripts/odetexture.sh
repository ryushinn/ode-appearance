#!/usr/bin/env bash
# Usage:
#   scripts/odetexture.sh <data_directory> <output_directory>
#
# Arguments:
#   $1  data_directory: The path to the directory containing the data.
#   $2  output_directory: The path to the directory where the output will be saved.

source scripts/data_texture.sh

for datum in "${data[@]}"; do
    echo "training ode-texture with data $datum"

    python main_odetexture.py --exp_path $2 --exemplars_path "$1/$datum.gif" --comment "$datum"

done
