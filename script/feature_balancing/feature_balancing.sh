#!/bin/bash
# Go to the root directory
cd ../../
# Define the path where the python files are
path_to_scripts='pipeline/feature-balancing/*'

for f in $path_to_scripts
do
    echo "Processing $f file..."
    ipython $f &
done

