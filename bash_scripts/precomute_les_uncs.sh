#!/bin/bash

source activate clseg

path_pred=""
path_save=""
do_probas=(0.01 0.05 0.1 0.15)
do_thresholds=(0.55 0.55 0.55 0.45)

for setname in "test_in" "train" "val" "train"; do
  echo "DE uncertainty"
  python precomputing/lesion_uncertainties_all.py \
  --path_pred "${path_pred}/de/predictions_${setname}_npz" \
   --set_name $setname \
   --path_save "${path_save}/de" \
   --n_jobs 19 \
   --proba_threshold 0.55 --l_min 2 --temperature 1.0

   for (( i=0; i<${#do_probas[@]}; i++ )); do
      echo "DP ${do_probas[i]} (threshold ${do_thresholds[i]}): ${setname} DSC-RC"
      python precomputing/lesion_uncertainties_all.py \
      --path_pred "${path_pred}/dropout_prob_${do_probas[i]}/predictions_${setname}_npz" \
       --set_name $setname \
       --path_save "${path_save}/dropout_prob_${do_probas[i]}" \
       --n_jobs 19 \
       --proba_threshold ${do_thresholds[i]} --l_min 2 --temperature 1.0
   done
done
