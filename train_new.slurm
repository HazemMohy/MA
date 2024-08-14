#!/bin/bash

#SBATCH -J 3D_Seg_UNET
#SBATCH --output=/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_%j_output.txt
#SBATCH --error=/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_%j_error.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2  # If your task can utilize 2 GPUs effectively
#SBATCH --mem=100G    # Reduced memory request
#SBATCH --cpus-per-task=10  # Adjusted based on available GPU resources
#SBATCH --time=2-00:00:00  # Keep if you expect the job to possibly run long
#SBATCH --qos=gpu_normal  # Ensure it matches the allowed QoS for gpu_p
#SBATCH --nice=1000  # Adjust as necessary

source /lustre/groups/iterm/Hazem/miniconda3/bin/activate Masterarbeit
eval "$(conda shell.bash hook)"
conda activate Masterarbeit
echo "env activated"

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING set to 1"

srun python -u Segmentation_3D_TL.py --job $SLURM_JOB_ID
