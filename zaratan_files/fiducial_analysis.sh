#!/bin/bash --login
#SBATCH -n 40                     #Number of processors in our pool
#SBATCH -t 2:00:00               #Max wall time for entire job

#change the partition to compute if running in Swansea
#SBATCH -p htc                    #Use the High Throughput partition which is intended for serial jobs

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
file_list=$(ls $directory/output_*)

# Run the tasks:
$parallel "$srun python3 $script {}" ::: $file_list
# in this case, we are running a script named runtask, and passing it a single argument
# {1} is the first argument
# parallel uses ::: to separate options. Here {1..64} is a shell expansion defining the values for
#    the first argument, but could be any shell command
#
# so parallel will run the runtask script for the numbers 1 through 64, with a max of 40 running 
#    at any one time
#
# as an example, the first job will be run like this:
#    srun -N1 -n1 --exclusive ./runtask arg1:1