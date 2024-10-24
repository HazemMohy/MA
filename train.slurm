#!/bin/bash

#SBATCH -c 16
#SBATCH -J 3D_Seg_UNET
#SBATCH --output=/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_%j_output.txt
#SBATCH --error=/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_%j_error.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1
#SBATCH --qos=gpu_normal
#SBATCH -w gpusrv26
##SBATCH --exclude=supergpu15
##SBATCH --exclude=gpusrv[XX,YY],supergpuZZ

source /lustre/groups/iterm/Hazem/miniconda3/bin/activate Masterarbeit
eval "$(conda shell.bash hook)"
conda activate Masterarbeit
echo "env activated"


# Set CUDA_LAUNCH_BLOCKING to 1 for better error traceability
export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING set to 1"

#srun python -u 1st_TL_Solutions/UNET3D_git_TL_2B_Rami_7.py --job $SLURM_JOB_ID
srun python -u UNET3D_git_TL_1E.py --job $SLURM_JOB_ID
#srun python -u hello.py --job $SLURM_JOB_ID
