#!/bin/bash

base=./logs/denoising_new
common_args="--correct_darkshading --correct_illum --visualize_img"

exp="sid_eld"
save_folder="results/$exp"
resume="pretrained_ckpts/${exp}_final.pth"

for dataset in SID ELD; do
  if [ "$dataset" == "SID" ]; then
    ratios=(100 250 300)
  else
    ratios=(100 200)
  fi

  for ratio in "${ratios[@]}"; do
    python test_denoising.py \
      --save_folder "$save_folder" \
      --resume "$resume" \
      --test_dataset "$dataset" \
      --ratio "$ratio" \
      $common_args
  done
done
