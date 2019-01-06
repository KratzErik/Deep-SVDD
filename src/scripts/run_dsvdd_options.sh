#!/usr/bin/env bash

dataset=$1
sh scripts/any_smile_dataset_svdd.sh $dataset gpu soft 0 adam 0.0001 150 1 0 1 0 sunny_highway 64 0 &&
sh scripts/any_smile_dataset_svdd.sh $dataset gpu oneclass 0 adam 0.0001 150 1 1 1 0 sunny_highway 64 0
