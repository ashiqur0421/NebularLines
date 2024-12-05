#!/bin/bash

# Specify the directory containing the files
directory="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial/" 

# Specify the files
file_format="output_*" 

# Specify the Python script to run
script="galaxy_emission.py" 

# Loop through all files
for file in "$directory"/"$file_format"; do
  # Run the Python script on each file
  python "$script" "$file"
done