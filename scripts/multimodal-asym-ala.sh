#!/bin/bash
python main.py \
--data wiki_std_synset \
--epochs 30 \
--output_mode multimodal \
--gpu 0  \
--w2v_type wiki_unique_std  \
--name multimodal-asym \
--cluster_type HAC \
--test_mode false \
--model_type asym \
--loss weight 
