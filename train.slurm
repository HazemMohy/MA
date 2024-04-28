#!/bin/bash

#SBATCH -c 16
#SBATCH -J 3D_Seg_UNET
#SBATCH --output=/lustre/groups/iterm/Hazem/MA/hpc/slurm_outputs/3D_Seg_%j_output.txt
#SBATCH --error=/lustre/groups/iterm/Hazem/MA/hpc/slurm_outputs/3D_Seg_%j_error.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal
##SBATCH -w supergpu02

eval "$(conda shell.bash hook)"
conda activate Masterarbeit
echo "env activated"

echo "Time Loacation"
which time

python -u UNET3D.py