#!/usr/bin/bash

cd ./egs2/ljspeech/tts1
. ./path.sh
. ./run.sh \
    --inference_nj 1 \
    --gpu_inference true \
    --stage 7 \
    --stop_stage 7 \
    --tts_stats_dir exp/tts_stats \
    --expdir exp/fastspeech_gw \
    --tts_exp exp/fastspeech_gw/gw_new_mexp \
    --train_config conf/tuning/train_fastspeech_gw.yaml \
    --train_args \
    "--wandb_name gw \
    --batch_bins 1500000 \
    --accum_grad 16 \
    " \
    --inference_tag 235epoch \
    --inference_model 235epoch.pth \
    --inference_args \
    "--use_teacher_forcing true \
    --ngpu 1\
    "
    # --vocoder_file pretrained_model/ljspeech_hifigan.v1/checkpoint-2500000steps.pkl
    
