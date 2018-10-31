#!/usr/bin/env bash

device=$1
xp_dir=../log/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
nu=$7
hard_margin=$8
center_fixed=$9
block_coordinate=${10}
architecture=${11}
rep_dim=${12}
n_train=${13}
n_test=${14}
in_name=${15}


mkdir $xp_dir;

# BDD100K training
python baseline.py --dataset bdd100k --solver $solver --loss svdd --lr $lr --lr_drop 1 --lr_drop_in_epoch 50 \
    --seed $seed --lr_drop_factor 10 --block_coordinate $block_coordinate --center_fixed $center_fixed \
    --use_batch_norm 1 --pretrain 0 --in_name $in_name --batch_size 200 --n_epochs $n_epochs --device $device \
    --xp_dir $xp_dir --leaky_relu 1 --weight_decay 1 --C 1e6 --reconstruction_penalty 0 --c_mean_init 1 \
    --hard_margin $hard_margin --nu $nu --out_frac 0 --weight_dict_init 0 --unit_norm_used l1 --gcn 1 --bdd100k_bias 0 \
     --nnet_diagnostics 0 --e1_diagnostics 0 --bdd100k_architecture $architecture --bdd100k_rep_dim $rep_dim ;


# Experiment config is mainly set in bdd100k part of config.py, but parameters that change a lot can be added in baseline.py and specified here for convenience.

# Run experiment with sh bdd100k_svdd.sh gpu bdd100k/your_folder_spec 0 adam 0.0001 150 1 0 1 0 2 100 100
