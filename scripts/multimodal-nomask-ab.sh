#!/bin/bash
# 用于堆mu进行分析
#
#array=( 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
#array=( 0.1 )
array=( 0.00 0.07 0.14 0.21 0.29 0.36 0.43 0.50 0.57 0.64 0.71 0.79 0.86 0.93 1.00 )
for mu in ${array[@]}
do
    echo $mu
    python main.py \
    --data wiki_std_synset \
    --epochs 16 \
    --output_mode multimodal \
    --gpu 2  \
    --w2v_type wiki_unique_std  \
    --name "multimodal-wTM-$mu" \
    --cluster_type HAC \
    --test_mode false \
    --model_type withoutTM \
    --loss weight \
    --mu $mu
    if ! command; then echo "command failed"; exit 1; fi
done;
