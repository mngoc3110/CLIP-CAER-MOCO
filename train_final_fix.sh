#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Final_MoCo_Steps4_Layers3 \
  --gpu 0 \
  --epochs 80 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0002 \
  --milestones 35 60 \
  --gamma 0.1 \
  --temporal-layers 3 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --gradient-accumulation-steps 4 \
