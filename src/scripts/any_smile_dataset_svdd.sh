#!/usr/bin/env bash

dataset=$1
device=$2
xp_dir=../log/prosivic/$3
seed=$4
solver=$5
lr=$6
n_epochs=$7
nu=$8
hard_margin=$9
center_fixed=${10}
block_coordinate=${11}
in_name=${12}
batch_size=${13}
weight_dict_init=${14}

mkdir -p $xp_dir;

# BDD100K training
python baseline.py --dataset $dataset --solver $solver --loss svdd --lr $lr --lr_drop 1 --lr_drop_in_epoch 50 \
    --seed $seed --lr_drop_factor 10 --block_coordinate $block_coordinate --center_fixed $center_fixed \
    --use_batch_norm 1 --pretrain 1 --batch_size $batch_size --n_epochs $n_epochs --device $device \
    --xp_dir $xp_dir --leaky_relu 1 --weight_decay 1 --C 1e6 --reconstruction_penalty 0 --c_mean_init 1 \
    --hard_margin $hard_margin --nu $nu --out_frac 0 --weight_dict_init $weight_dict_init --unit_norm_used l1 --gcn 1 --bias 0 \
     --nnet_diagnostics 1 --e1_diagnostics 0 ;


# Experiment config is mainly set in dataset specific part of config.py, but parameters that change a lot can be added in baseline.py and specified here for convenience.

# Run experiment with sh scripts/any_smile_dataset_svdd.sh dataset gpu experiment_specific_folder 0 adam 0.0001 50 1 0 0 1 sunny_highway 64 0
