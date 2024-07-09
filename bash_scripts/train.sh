#!/bin/bash

conda activate clseg

path_data=""
path_save=""

for seed in 1 2 3 ... 10; do
	python model_training/train.py \
	--exp_name "seed${seed}" --seed $seed \
	--model unet_shallow \
	--loss focal_cl --scheduler plateau --gamma 0.5 --step_size 10 --learning_rate 0.01 --warmup_iters 400 --learning_rate_min 1e-7 \
	--tolerance 1e-10 --patience 100 --best_metric "nDSC_1" --best_min_max "max" --n_epochs 400 \
	--input_modalities mp2rage \
	--input_train_paths "${path_data}/train/MP2RAGE_skull_stripped" \
	--target_train_path "${path_data}/train/gt_cl" \
	--input_val_paths "${path_data}/val/MP2RAGE_skull_stripped" \
	--target_val_path "${path_data}/val/gt_cl" \
	--target_prefix cl_binary.nii.gz --input_prefixes UNIT1.nii.gz \
	--n_patches 32 --num_workers 12 --batch_size 8 --cache_rate 0.3 --val_interval 1 \
	--path_save "${path_save}"
done

for proba in 0.01 0.05 0.1 0.15 0.2 0.25; do
    python model_training/train.py \
    --exp_name "dropout_prob_${proba}" --seed 0 \
    --model unet_shallow_dropout --dropout_proba $proba \
    --loss focal_cl --scheduler plateau --gamma 0.5 --step_size 10 --learning_rate 0.01 --warmup_iters 400 --learning_rate_min 1e-7 \
    --tolerance 1e-10 --patience 100 --best_metric "nDSC_1" --best_min_max "max" --n_epochs 400 \
    --input_modalities mp2rage \
	--input_train_paths "${path_data}/train/MP2RAGE_skull_stripped" \
	--target_train_path "${path_data}/train/gt_cl" \
	--input_val_paths "${path_data}/val/MP2RAGE_skull_stripped" \
	--target_val_path "${path_data}/val/gt_cl" \
    --target_prefix cl_binary.nii.gz --input_prefixes UNIT1.nii.gz \
    --n_patches 32 --num_workers 10 --batch_size 8 --cache_rate 0.3 --val_interval 1 \
    --path_save "${path_save}"
done