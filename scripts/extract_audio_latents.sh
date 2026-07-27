## split audio clips
DATA_DIR=# TODO
PATH_TO_AUDIO_DIR=$DATA_DIR/output_cut     # dir to audio clips e.g.: /home/to/audiocaps_wav
OUTPUT_PARTITION_FILE=$DATA_DIR/partition-file.tsv     # ouput csv path, e.g.: /home/to/output/audiocaps-test-partition.tsv

#python training/partition_clips.py \
#    --data_dir $PATH_TO_AUDIO_DIR \
#    --output_dir $OUTPUT_PARTITION_FILE


## extract audio latents
#export CUDA_VISIBLE_DEVICES=0

CAPTIONS_TSV=$DATA_DIR/yt8m-captions.tsv     # captions tsv path, e.g.: /home/to/audiocaps-test.tsv
OUTPUT_LATENT_DIR=$DATA_DIR/test-latent     # output latent dir, e.g.: /home/to/output/audiocaps-test-latent
OUTPUT_NPZ_DIR=$DATA_DIR/musiccaps-test-npz     # output npz dir, e.g.: /home/to/output/audiocaps-test-npz

torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \
    --captions_tsv $CAPTIONS_TSV \
    --data_dir $PATH_TO_AUDIO_DIR \
    --clips_tsv $OUTPUT_PARTITION_FILE \
    --latent_dir $OUTPUT_LATENT_DIR \
    --output_dir $OUTPUT_NPZ_DIR \
    --text_encoder='t5_clap'  # ['clip', 't5', 't5_clap']
