#!/bin/bash --login
#SBATCH -n 40                     #Number of processors
#SBATCH -t 2:00:00               #Max wall time for entire job

module purge
module load parallel
module load python
python -m pip install --upgrade pip
python -m pip install --user yt
python -m pip install --user "yt[ramses]"
pip install numpy
pip install matplotlib
pip install astropy
pip install scipy

# Define srun arguments:
srun="srun -n1 -N1 --exclusive"
# --exclusive     ensures srun uses distinct CPUs for each job step
# -N1 -n1         allocates a single core to each task

# Define parallel arguments:
parallel="parallel -N 1 --delay .2 -j $SLURM_NTASKS --joblog parallel_joblog --resume"
# -N 1              is number of arguments to pass to each job
# --delay .2        prevents overloading the controlling node on short jobs
# -j $SLURM_NTASKS  is the number of concurrent tasks parallel runs, so number of CPUs allocated
# --joblog name     parallel's log file of tasks it has run
# --resume          parallel can use a joblog and this to continue an interrupted run (job resubmitted)

# Specify the directory containing the files
directory="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial" 

# Specify the files
file_format="output_*" 

# Specify the Python script to run
script="zaratan_files/galaxy_emission.py" 

#file_list="$directory/output_00304 $directory/output_00305"
dir_list=$(ls -d $directory/output_*)

# Run the tasks:
$parallel "$srun python3 $script {}" ::: $dir_list  
# in this case, we are running a script and passing it a single argument
# parallel uses ::: to separate options.
