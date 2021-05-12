#!/bin/bash
python main.py \
--data new_text_data \
--epochs 30 \
--output_mode multimodal \
--gpu 0  \
--w2v_type mge_w2v  \
--name multimodal-asym-tri \
--cluster_type HAC \
--test_mode false \
--model_type asym \
--loss triplet
