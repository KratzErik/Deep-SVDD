#!/usr/bin/env bash

device=$1
xp_dir=../log/prosivic/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
nu=$7
hard_margin=$8
center_fixed=$9
block_coordinate=${10}
in_name=${11}
batch_size=${12}
weight_dict_init=${13}

mkdir $xp_dir;

# BDD100K training
python baseline.py --dataset prosivic --solver $solver --loss svdd --lr $lr --lr_drop 1 --lr_drop_in_epoch 50 \
    --seed $seed --lr_drop_factor 10 --block_coordinate $block_coordinate --center_fixed $center_fixed \
    --use_batch_norm 1 --pretrain 1 --batch_size $batch_size --n_epochs $n_epochs --device $device \
    --xp_dir $xp_dir --leaky_relu 1 --weight_decay 1 --C 1e6 --reconstruction_penalty 0 --c_mean_init 1 \
    --hard_margin $hard_margin --nu $nu --out_frac 0 --weight_dict_init $weight_dict_init --unit_norm_used l1 --gcn 1 --dreyeve_bias 0 \
     --nnet_diagnostics 1 --e1_diagnostics 1 ;


# Experiment config is mainly set in dataset specific part of config.py, but parameters that change a lot can be added in baseline.py and specified here for convenience.

# Run experiment with sh prosivic_svdd.sh gpu prosivic/experiment_specific_folder 0 adam 0.0001 150 1 0 1 0 inlier_class_name 100 0
