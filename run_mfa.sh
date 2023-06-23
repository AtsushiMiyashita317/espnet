#!/usr/bin/bash

cd ./egs2/ljspeech/tts1
. ./path.sh
. ./run.sh --stage 6 \
    --stop_stage 6 \
    --inference_nj 8 \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn\
    --test_sets "dev_phn eval1_phn tr_part_phn" \
    --cleaner none \
    --g2p none \
    --train_config conf/tuning/train_fastspeech2.yaml \
    --train_args \
    "\
    --use_wandb true \
    --wandb_name gw_mfa_nearest \
    --batch_bins 1500000 \
    --accum_grad 16 \
    --tts_conf lr_before=false \
    "\
    --teacher_dumpdir data \
    --tts_stats_dir exp/mfa_stats \
    --expdir exp/fastspeech2 \
    --tts_exp exp/fastspeech2/gw_mfa_nearest \
    --write_collected_feats true \
    --srctexts data/local/mfa/text \
    --inference_model valid.loss.best.pth \
    --inference_tag valid_loss_best \
    --inference_args \
    "--use_teacher_forcing true"    

    