#!/bin/bash
#SBATCH --job-name=32proc
#SBATCH --exclusive
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=32
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err


source /home/apps/MSCC/miniconda3/bin/activate
conda activate mscc
cd $SLURM_SUBMIT_DIR
mpirun -np 32 python exe.py input_chain18_singlet_np4.in
