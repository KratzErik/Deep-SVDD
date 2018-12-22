#!/usr/bin/env bash

dataset=$1
exp_name=$2
epochs=$3
opt1="_101"
opt2="_010"
opt3="_110"
opt4="_001"
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt1 0 adam 0.0001 $epochs 1 1 0 1 sunny_highway 64 0 &&
cp -r ../log/$dataset/$exp_name$opt1/ ../log/$dataset/$exp_name$opt2/
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt2 0 adam 0.0001 $epochs 1 0 1 0 sunny_highway 64 0 &&
cp -r ../log/$dataset/$exp_name$opt1/ ../log/$dataset/$exp_name$opt3/
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt3 0 adam 0.0001 $epochs 1 1 1 0 sunny_highway 64 0
cp -r ../log/$dataset/$exp_name$opt1/ ../log/$dataset/$exp_name$opt4/
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt4 0 adam 0.0001 $epochs 1 0 0 1 sunny_highway 64 0
