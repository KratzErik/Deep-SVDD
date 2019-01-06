#!/usr/bin/env bash
#sh scripts/any_smile_dataset_svdd.sh prosivic gpu oneclass_c_fixed 0 adam 0.0001 50 1 0 0 1 sunny_highway 64 0 &&
#sh scripts/any_smile_dataset_svdd.sh prosivic gpu soft_c_fixed 0 adam 0.0001 50 1 0 0 1 sunny_highway 64 0 &&
sh scripts/any_smile_dataset_svdd.sh dreyeve gpu oneclass_c_fixed 0 adam 0.0001 50 1 0 0 1 sunny_highway 64 0 &&
sh scripts/any_smile_dataset_svdd.sh dreyeve gpu soft_c_fixed 0 adam 0.0001 50 1 0 0 1 sunny_highway 64 0

