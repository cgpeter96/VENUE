python main_new.py \
                --data new_text_data \
                --epochs 20 \
                --output_mode multimodal \
                --gpu 0  \
                --w2v_type mge_w2v  \
                --name multimodal_asym_param-1024-1024 \
                --cluster_type HAC \
                --test_mode false \
                --model_type asym \
                --loss weight \
                --mu 0.3 \
                --compact_dim 1024 \
                --output_dim 1024
            

