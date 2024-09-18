#!/bin/bash


#SBATCH --job-name=no_lora_final
#SBATCH --partition=student_project
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/slurm/%x_%j.out    # Standard output
#SBATCH --error=./logs/slurm/%x_%j.out     # Standard error

mkdir -p ./logs/slurm

# Activate conda environment
source /opt/modules/i12g/anaconda/3-2019.10/etc/profile.d/conda.sh
conda activate regulate-me


# Print GPU status
# echo "GPU Status:"
# nvidia-smi

# Run your Python script
srun python /s/project/ml4rg_students/2024/Project07_PolyB/Regulate-Me/scripts/run_sweep.py