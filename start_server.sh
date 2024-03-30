#!/bin/bash

/root/miniconda3/envs/G2G/bin/python server.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --intermediate_dir intermediate/ \
  --checkpoint_dir checkpoint \
  --checkpoint model_best.pt \
  --known_class False \
  --beam_size 10 \
  --stepwise False \
  --use_template False