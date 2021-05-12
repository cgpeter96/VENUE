#!/bin/bash
python main.py \
--data wiki_std_synset \
--epochs 20 \
--output_mode multimodal \
--gpu 1  \
--w2v_type wiki_unique_std  \
--name multimodal-full \
--cluster_type HAC \
--test_mode false \
--model_type full \
--loss weight 
