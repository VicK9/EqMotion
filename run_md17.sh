#!/bin/bash
module purge
module load 2021
module load Anaconda3/2021.05
# source start_venv.sh
source activate locs_new
# # Take input from command line the flag to use transformer or not
INPUT_ARGS=$@

GPU=0 # Set to whatever GPU you want to use

# Make sure to replace this with the directory containing the data files
DATA_PATH='data/md17/'

BASE_RESULTS_DIR="results/md17"
EXPERIMENT_EXT=""

WINDOW_SIZE=10
SEED=1
MODEL_TYPE="locs"

echo "Running model: ${MODEL_TYPE}_${EXPERIMENT_EXT}_seed-${SEED}_window-${WINDOW_SIZE}"
WORKING_DIR="${BASE_RESULTS_DIR}/${MODEL_TYPE}_${EXPERIMENT_EXT}/seed_${SEED}/"

GNN_ARGS="--gnn_hidden_size 64 --gnn_n_layers 4 --gnn_dropout 0 --gnn_n_in_dims 6 --gnn_n_out_dims 3"
LOCALIZER_ARGS="--localizer_embedding_size 32 --localizer_hidden_size 64 --localizer_n_layers 2 --localizer_dropout 0"
MODEL_ARGS="--model_type ${MODEL_TYPE} $GNN_ARGS $LOCALIZER_ARGS --seed ${SEED} --window_size ${WINDOW_SIZE}"
~/.conda/envs/locs_new/bin/python  -u main_md17.py $INPUT_ARGS $MODEL_ARGS 

      