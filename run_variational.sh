#!/usr/bin/bash

cd ./egs2/ljspeech/tts1
. ./path.sh
. ./run.sh \
    --stage 7 \
    --stop_stage 7 \
    --inference_nj 16 \
    --tts_stats_dir exp/tts_stats \
    --expdir exp/fastspeech_gw \
    --tts_exp exp/fastspeech_gw/test_ode \
    --train_config conf/tuning/train_variational_fastspeech_gw.yaml \
    --train_args \
    "\
    --wandb_name test_ode \
    --batch_bins 1500000 \
    --accum_grad 16 \
    " \
    --inference_tag valid_loss_best \
    --inference_model valid.loss.best.pth \
    --inference_args \
    "--use_teacher_forcing true \
    --ngpu 1"    
