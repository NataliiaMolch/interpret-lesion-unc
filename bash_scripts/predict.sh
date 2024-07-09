#!/bin/bash
conda activate clseg
path_data="" # folder containing a set_name folder with the data
set_name="test" # "train" | "val"
path_save=""
ensemble_checkpoints=() # a list with full paths to the model checkpoints .pth files
model_checkpoint="" # path to the model checkpoints .pth file

echo "DE model inference"
python model_evaluation/save_ens_pred_npz.py \
--exp_name "dropout_prob_${proba}" --eval_set_name "${set_name}" --seed 0 \
--model_type "de" \
--model "unet_shallow" \
--ensemble_checkpoints "$ensemble_checkpoints_arg" \
--input_modalities "mp2rage" \
--activation "none" \
--model_checkpoint "${exp_dir}/mcdp/dropout_prob_${proba}/model_epoch_${do_epochs[i]}.pth" \
--input_modalities "mp2rage" \
--input_val_paths "${path_data}/${set_name}/MP2RAGE_skull_stripped" --input_prefixes "UNIT1.nii.gz" \
--target_val_path "${path_data}/${set_name}/gt_cl" --target_prefix "cl_binary.nii.gz" \
--bm_val_path "${path_data}/${set_name}/brain_mask" --bm_prefix "UNIT1_brain_mask_cc.nii.gz" \
--path_save "${path_save}" \
--num_workers 10

echo "MCDP model inference"
for (( i=0; i<${#do_probas[@]}; i++ )); do
    python model_evaluation/save_ens_pred_npz.py \
    --exp_name "dropout_prob_${do_probas[i]}" --eval_set_name "${set_name}" --seed 0 \
    --model_type "mcdp" --n_samples 10 \
    --model "unet_shallow_dropout" --dropout_proba ${do_probas[i]} \
    --activation "none" \
    --model_checkpoint "${exp_dir}/mcdp/dropout_prob_${do_probas[i]}/model_epoch_${do_epochs[i]}.pth" \ \
    --input_modalities "mp2rage" \
    --input_val_paths "${path_data}/${set_name}/MP2RAGE_skull_stripped" --input_prefixes "UNIT1.nii.gz" \
    --target_val_path "${path_data}/${set_name}/gt_cl" --target_prefix "cl_binary.nii.gz" \
    --bm_val_path "${path_data}/${set_name}/brain_mask" --bm_prefix "UNIT1_brain_mask_cc.nii.gz" \
    --path_save "${path_save}" \
    --num_workers 10
done