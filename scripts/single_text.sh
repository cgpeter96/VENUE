#!/bin/bash
python main.py \
--data wiki_std_synset \
--epochs 30 \
--output_mode text \
--gpu 1  \
--w2v_type wiki_unique_std  \
--name singe_text_tanh \
--cluster_type HAC \
--test_mode false 
