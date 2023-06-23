#!/usr/bin/bash

cd ./egs2/ljspeech/tts1
. ./path.sh
. ./run.sh --stage 7 \
    --stop_stage 7 \
    --inference_nj 4 \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn\
    --test_sets "dev_phn eval1_phn" \
    --cleaner none \
    --g2p none \
    --train_config conf/tuning/train_vnart.yaml \
    --train_args \
    "\
    --use_wandb true \
    --wandb_name vnart_no_dropout \
    "\
    --teacher_dumpdir data \
    --tts_stats_dir exp/mfa_stats \
    --expdir exp/vnart \
    --tts_exp exp/vnart/no_dropout \
    --write_collected_feats true \
    --srctexts data/local/mfa/text \
    --inference_model 219epoch.pth \
    --inference_tag 219epoch \
    --inference_args \
    "--use_teacher_forcing false \
    --ngpu 0"    
