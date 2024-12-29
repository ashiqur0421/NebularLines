#!/bin/bash

#SBATCH --ntasks=6
#SBATCH -t 1:0:0

export SLURM_EXPORT_ENV=ALL

module purge
module load python
python -m pip install --upgrade pip
python -m pip install --user yt
python -m pip install --user "yt[ramses]"
pip install numpy
pip install matplotlib
pip install astropy
pip install scipy

# Specify the directory containing the files
directory="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial" 

# Specify the files
file_format="output_*" 

# Specify the Python script to run
script="zaratan_files/galaxy_emission.py" 

# Loop through all files
#for file in $directory/output_*; do
  # Run the Python script on each file
#  python3 $script "$file" & 
#done

srun python3 $script "$directory/output_00304" &
srun python3 $script "$directory/output_00305" &
srun python3 $script "$directory/output_00306" &
srun python3 $script "$directory/output_00308" &
srun python3 $script "$directory/output_00310" &
srun python3 $script "$directory/output_00311" &

wait


