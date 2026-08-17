#!/bin/bash

INPUT_DIR="$SCRIPT_DIR/../docs/time_course/input"
OUTPUT_DIR="$SCRIPT_DIR/../docs/time_course/results"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*_P1_Day*.tiff
do
    filename=$(basename "$file")

    # Cellline_P1_Day3.tiff -> Cellline
    cellline=$(echo "$filename" | sed -E 's/_P1_Day[0-9]+\.tiff$//')

    # Cellline_P1_Day3.tiff -> 3
    day=$(echo "$filename" | sed -E 's/.*_P1_Day([0-9]+)\.tiff/\1/')

    echo "========================================"
    echo "Cell line: $cellline"
    echo "Day:       $day"
    echo "Input:     $file"
    echo "========================================"

    mkdir -p "$OUTPUT_DIR/$cellline/Day$day"

    bash ../../Morphi-models/scripts/run_full_pipeline.sh \
        --input "$file" \
        --output "$OUTPUT_DIR/$cellline/Day$day"
done