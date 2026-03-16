#!/bin/bash

base=./logs/denoising_new
common_args="--correct_illum --visualize_img"

exp="lrid"
save_folder="results/$exp"
resume="pretrained_ckpts/${exp}_final.pth"

for condition in indoor outdoor; do
  if [ "$condition" == "indoor" ]; then
    ratios=(1 2 4 8 16)
  else
    ratios=(1 2 4)
  fi

  for ratio in "${ratios[@]}"; do
    python test_LRID_denoising.py \
      --save_folder "$save_folder" \
      --resume "$resume" \
      --condition "$condition" \
      --ratio "$ratio" \
      $common_args
  done
done