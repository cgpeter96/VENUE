#!/bin/bash
param=(128 256 512 1024)
for((i=0;i<${#param[@]};i++))
    do
    # echo ${param[]}
    for((j=0;j<${#param[@]};j++))
        do  
            echo ${param[i]}"-"${param[j]}
            python main.py \
                --data wiki_std_synset \
                --epochs 20 \
                --output_mode multimodal \
                --gpu 1  \
                --w2v_type wiki_unique_std  \
                --name multimodal-"${param[i]}"-"${param[j]}" \
                --cluster_type HAC \
                --test_mode false \
                --model_type full \
                --loss weight \
                --mu 0.3 \
                --compact_dim ${param[i]} \
                --output_dim ${param[j]}
            
        done
    done

