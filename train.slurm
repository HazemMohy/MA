#!/bin/bash

## 16 core CPU
#SBATCH -c 16

## Job Name
#SBATCH -J train_job


#SBATCH --output=o_%j.txt
#SBATCH --error=e_%j.txt
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=160G 
#SBATCH --time=2-00:00:00
#SBATCH --nice=1000
#SBATCH --qos=gpu_normal
##SBATCH -w supergpu02