#!/bin/sh
#SBATCH -N 1 
#SBATCH --time=4-00:00 
#SBATCH --job-name=fibonacci  %change name of ur job
#SBATCH --error=error   %change name the error file
#SBATCH --output=output  %change name of ur output file
#SBATCH --partition=gpu  %there are various partition. U can change various GPUs
#SBATCH --gres=gpu:2 %same as above
cd $SLURM_SUBMIT_DIR


source /home/rajeshr.scee.iitmandi/anaconda3/bin/activate tf_gpu
python /home/rajeshr.scee.iitmandi/fibonacci.py

