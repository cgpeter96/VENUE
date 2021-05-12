#!/bin/bash
python main.py \
--data wiki_std_synset \
--epochs 25 \
--output_mode visual \
--gpu 1  \
--w2v_type wiki_unique_std  \
--name singe_visual_mean_before \
--cluster_type HAC \
--test_mode false 
