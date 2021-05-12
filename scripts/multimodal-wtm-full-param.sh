#!/bin/bash
param=(128 256 512 1024)
for((i=0;i<${#param[@]};i++))
    do
    # echo ${param[]}
    for((j=0;j<${#param[@]};j++))
        do  
            echo ${param[i]}"-"${param[j]}
            python main_new.py \
                --data new_text_data \
                --epochs 20 \
                --output_mode multimodal \
                --gpu 2  \
                --w2v_type mge_w2v  \
                --name multimodal_asym_param-"${param[i]}"-"${param[j]}" \
                --cluster_type HAC \
                --test_mode false \
                --model_type asym \
                --loss weight \
                --mu 0.3 \
                --compact_dim ${param[i]} \
                --output_dim ${param[j]}
            
        done
    done

