#!/usr/bin/env bash

declare -a datasets=( \
  "ACRIM1_HF_full2.txt" \
  "ACRIM2_ERBS_full.txt" \
  "MultiSatellites_HF_ERBSmultiply1.txt" \
  "VIRGOFused_ACRIM2_ERBS.txt" \
  "VIRGOFused_ACRIM3_ERBS.txt" \
  "VIRGOFused_ACRIM3_TIM0.txt" \
  "VIRGOFused_TIM.txt" \
  "VIRGOFused_TIM_PREMOSFused.txt" \
  "VIRGOFused_TISIS.txt"
)

N=${#datasets[@]}

DATA_DIR="/home/rok/tsipy/data/tsi_40"

FUSION_MODEL="localgp"
NUM_INDUCING_PTS=500
MAX_ITER=5000
PRED_WINDOW=1.0
FIT_WINDOW=3.0

for (( i=0; i<${N}; i++ ));
do
  DATASET=${datasets[$i]}

  echo -e "python exp_tsi_40_dataset.py\n" \
        "--experiment_name=exp_tsi_40_datasets\n" \
        "--data_dir=${DATA_DIR}\n" \
        "--dataset_file=${DATASET}\n" \
        "--fusion_model=${FUSION_MODEL}\n" \
        "--num_inducing_pts=${NUM_INDUCING_PTS}\n" \
        "--max_iter=${MAX_ITER}\n" \
        "--pred_window=${PRED_WINDOW}\n" \
        "--fit_window=${FIT_WINDOW}\n"

  python exp_tsi_40_dataset.py \
        --experiment_name=exp_tsi_40_datasets \
        --data_dir=${DATA_DIR} \
        --dataset_file=${DATASET} \
        --fusion_model=${FUSION_MODEL} \
        --num_inducing_pts=${NUM_INDUCING_PTS} \
        --max_iter=${MAX_ITER} \
        --pred_window=${PRED_WINDOW} \
        --fit_window=${FIT_WINDOW}

  echo -e "\n\n"
done
