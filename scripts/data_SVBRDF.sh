#!/usr/bin/env bash
# The list of dynamic SVBRDF data.

data=(
    "synthetic/metal_rusting"
    "synthetic/paintedmetal_rusting"
    "synthetic/leather_aging"
    "synthetic/wood_aging"
    "synthetic/sand_drying"
    "synthetic/ceramics_dirtcovering"
    "synthetic/cement_leveling"
    "paper_drying"
    "ice_melting"
    "salt_crystallizing"
    "salt_crystallizing_2"
    "copper_crystallizing"
    "honeycomb_melting"
    "honeycomb_melting_2"
    "watercolor_painting"
    "clay_solidifying"
    "copper_patinating"
    "copper_patinating_2"
    "steak_cooking"
    "cheese_melting"
    "cress_germinating"
)

echo "loading data: ${data[*]}"
