#!/bin/bash

set -e

## split audio clips
DATASET=dev
DATA_DIR=data/raw/$DATASET/
OUTPUT_DIR=data/meanaudio/$DATASET
mkdir -p $OUTPUT_DIR
PATH_TO_AUDIO_DIR=$DATA_DIR/train_data     # dir to audio clips e.g.: /home/to/audiocaps_wav
OUTPUT_PARTITION_FILE=$OUTPUT_DIR/devset-train-partition.tsv     # ouput csv path, e.g.: /home/to/output/audiocaps-test-partition.tsv

python training/partition_clips.py \
    --data_dir $PATH_TO_AUDIO_DIR \
    --output_dir $OUTPUT_PARTITION_FILE


## extract audio latents
export CUDA_VISIBLE_DEVICES=9

CAPTIONS_TSV=$DATA_DIR/devset_train.tsv     # captions tsv path, e.g.: /home/to/audiocaps-test.tsv
OUTPUT_LATENT_DIR=$OUTPUT_DIR/devset-train-latent     # output latent dir, e.g.: /home/to/output/audiocaps-test-latent
OUTPUT_NPZ_DIR=$OUTPUT_DIR/devset-train-npz     # output npz dir, e.g.: /home/to/output/audiocaps-test-npz

torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \
    --captions_tsv $CAPTIONS_TSV \
    --data_dir $PATH_TO_AUDIO_DIR \
    --clips_tsv $OUTPUT_PARTITION_FILE \
    --latent_dir $OUTPUT_LATENT_DIR \
    --output_dir $OUTPUT_NPZ_DIR \
    --text_encoder='t5_clap'  # ['clip', 't5', 't5_clap']

# If this is a validation or test dataset, make sure to extract the text embeddings, e.g.
# python extract_text.py --text_csv <captions file> --output_cache_path <gt cache location>

# If this is a validation set, make sure to prepare the gt_cache for av-bench, e.g.
# python evaluate.py --gt_audio <val data> --gt_cache <gt cache location> --audio_length 10 --skip_video_related --pred_cache <needs to be defined, can be any directory which is not a valid cache>