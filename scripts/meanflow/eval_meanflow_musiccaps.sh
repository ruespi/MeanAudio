#!/bin/bash

set -e

# evaluation on musiccaps

#export CUDA_VISIBLE_DEVICES=0

num_steps=1
ckpt_path=./weights/meanaudio_s_full.pth
#output_path=./exps/meanaudio_s_full/test_${num_steps}nfe_fp32_260731_235306
output_path=./exps/meanaudio_s_full/test_${num_steps}nfe_fp32_$(date +%y%m%d_%H%M%S)

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$ckpt_path" \
    --output $output_path/audio \
    --cfg_strength 0.9 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512 \
    --num_steps $num_steps \
    --use_meanflow \
    --full_precision \
    --eval_captions ./data/raw/musiccaps/musiccaps-captions-full.tsv

gt_audio=./data/raw/musiccaps/music_data  # not used if you specify gt_cache
gt_cache=./data/meanaudio/musiccaps/gt_cache_eval_full

pred_audio=$output_path/audio

source av-benchmark/.venv/bin/activate

python av-benchmark/evaluate.py \
    --gt_audio $gt_audio \
    --gt_cache $gt_cache \
    --pred_audio $pred_audio \
    --pred_cache $output_path/pred_cache \
    --audio_length=10 \
    --skip_video_related \
    --output_metrics_dir=$output_path \
    # --recompute_pred_cache \
    # --recompute_gt_cache
    # --debug

