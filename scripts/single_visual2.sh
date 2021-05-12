# 注意修改visualbackbone中 mean的位置
#!/bin/bash
python main.py \
--data wiki_std_synset \
--epochs 20 \
--output_mode visual \
--gpu 3  \
--w2v_type wiki_unique_std  \
--name singe_visual_mean_behind \
--cluster_type HAC \
--test_mode false 
