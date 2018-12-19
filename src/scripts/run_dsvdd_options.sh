#!/usr/bin/env bash

dataset=$1
exp_name=$2
opt1="_101"
opt2="_010"
opt3="_110"
opt4="_001"
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt1 0 adam 0.0001 100 1 1 0 1 sunny_highway 64 0 &&
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt2 0 adam 0.0001 100 1 0 1 0 sunny_highway 64 0 &&
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt3 0 adam 0.0001 100 1 1 1 0 sunny_highway 64 0
sh scripts/any_smile_dataset_svdd.sh $dataset gpu $exp_name$opt4 0 adam 0.0001 100 1 0 0 1 sunny_highway 64 0
