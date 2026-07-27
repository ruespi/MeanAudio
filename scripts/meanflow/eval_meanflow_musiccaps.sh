# evaluation on audiocaps

# export CUDA_VISIBLE_DEVICES=0

num_steps=1
ckpt_path=./weights/meanaudio_s_full.pth
output_path=./exps/meanaudio_s_full/test_${num_steps}nfe_fp32

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
    --full_precision

gt_audio=./data/musiccaps/wavs_eval  # not used if you specify gt_cache
gt_cache=./data/musiccaps/gt_cache_eval

pred_audio=$output_path/audio

python av-benchmark/evaluate.py \
    --gt_audio $gt_audio \
    --gt_cache $gt_cache \
    --pred_audio $pred_audio \
    --pred_cache $output_path/pred_cache \
    --recompute_pred_cache \
    --audio_length=10 \
    --skip_video_related \
    --output_metrics_dir=$output_path \
    # --recompute_gt_cache
    # --debug
