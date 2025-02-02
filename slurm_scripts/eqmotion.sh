#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=Eqmotion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --mem=120000M
#SBATCH --output=/home/vkyriacou88/EqMotion/slurm_scripts/outputs/eqmotion_%A.out
#SBATCH --array=1-4

module purge
module load 2021
module load Anaconda3/2021.05
# source start_venv.sh
source activate locs_new

# start in slurm_scripts
cd ~/EqMotion

# Define an array of the molecules to be used
molecules=("aspirin" "benzene" "ethanol" "malonaldehyde")  # Adjust script names as per your files

mol=${molecules[$SLURM_ARRAY_TASK_ID-1]}
 
echo "Running script ./run_md17.sh --wandb --localizer_type spatio_temporal --mol $mol"
# run the script
~/.conda/envs/locs_new/bin/python  -u main_md17.py --seed 555 --wandb --model_type eqmotion --mol $mol --epochs 300 --batch_size 50