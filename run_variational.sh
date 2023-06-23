#!/usr/bin/bash

cd ./egs2/ljspeech/tts1
. ./path.sh
. ./run.sh \
    --stage 7 \
    --stop_stage 7 \
    --inference_nj 1 \
    --tts_stats_dir exp/tts_stats \
    --expdir exp/fastspeech_gw \
    --tts_exp exp/fastspeech_gw/variational_weak_filter \
    --train_config conf/tuning/train_variational_fastspeech_gw.yaml \
    --train_args \
    "\
    --wandb_name variational \
    --batch_bins 1500000 \
    --accum_grad 16 \
    " \
    --inference_tag 340epoch \
    --inference_model 340epoch.pth \
    --inference_args \
    "--use_teacher_forcing true \
    --ngpu 1"    
