#!/usr/bin/env bash
# Usage:
#   scripts/sample_odetexture.sh <data_directory> <checkpoint_directory>
#
# Arguments:
#   $1  data_directory: The path to the directory containing the data.
#   $2  checkpoint_directory: The path to the directory where the trained ODEs was saved.

source scripts/data_texture.sh

for datum in "${data[@]}"; do
    echo "sampling with data $datum"

    python sample_odetexture.py --exp_path $2 --exemplars_path "$1/$datum.gif" --checkpoint_path "$2/$datum" --comment "sampling_256_$datum" --sample_size 256
    python sample_odetexture.py --exp_path $2 --exemplars_path "$1/$datum.gif" --checkpoint_path "$2/$datum" --comment "sampling_128_$datum" --sample_size 128

done
