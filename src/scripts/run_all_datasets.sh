#!/usr/bin/env bash
exp_name=$1
epochs=$2

sh scripts/any_smile_dataset_svdd.sh prosivic gpu $exp_name 0 adam 0.0001 $epochs 1 0 0 1 void 64 0 &
sh scripts/any_smile_dataset_svdd.sh dreyeve gpu $exp_name 0 adam 0.0001 $epochs 1 0 0 1 void 64 0

