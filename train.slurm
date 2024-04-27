#!/bin/bash

#SBATCH -c 16
<<<<<<< HEAD

## Job Name
#SBATCH -J 3D_Seg_UNET

#SBATCH --output=/home/iterm/hazem.abdel-rehim/MA/slurm_outputs/3D_Seg_o_%j.txt
#SBATCH --error=/home/iterm/hazem.abdel-rehim/MA/slurm_outputs/3D_Seg_e_%j.txt
=======
#SBATCH -J train_job
#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
>>>>>>> dcb202d2204f16b609dad6ef859c939ab2ac224f
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