#$ -l mem=10G
#$ -pe smp 8
#$ -l h_rt=2:00:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/ucabzc9/Scratch/
#$ -j y
#$ -N mmd_cubature_house

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

# Running date and nvidia-smi is useful to get some info in case the job crashes.

#module -f unload compilers mpi gcc-libs
#module load beta-modules
#module load gcc-libs/10.2.0
#module load cuda/11.2.0/gnu-10.2.0

## Load conda
module -f unload compilers
module load compilers/gnu/4.9.2
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate mmd_cubature

date

## Check if the environment is correct.
which pip
which python

python /home/ucabzc9/Scratch/mmd_flow_cubature/kh.py $JOB_PARAMS