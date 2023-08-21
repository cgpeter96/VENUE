# A Visually Enhanced Neural Encoder for Synset Induction

## Abstract
The synset induction task is to automatically cluster semantically identical instances, which are often represented by texts and images. Previous works mainly consider textual parts, while ignoring the visual counterparts. However, how to effectively employ the visual information to enhance the semantic representation for the synset induction is challenging. In this paper, we propose a Visually Enhanced NeUral Encoder (i.e., VENUE) to learn a multimodal representation for the synset induction task. The key insight lies in how to construct multimodal representations through intra-modal and inter-modal interactions among images and text. Specifically, we first design the visual interaction module through the attention mechanism to capture the correlation among images. To obtain the multi-granularity textual representations, we fuse the pre-trained tags and word embeddings. Second, we design a masking module to filter out weakly relevant visual information. Third, we present a gating module to adaptively regulate the modalities’ contributions to semantics. A triplet loss is adopted to train the VENUE encoder for learning discriminative multimodal representations. Then, we perform clustering algorithms on the obtained representations to induce synsets. To verify our approach, we collect a multimodal dataset, i.e., MMAI-Synset, and conduct extensive experiments. The experimental results demonstrate that our method outperforms strong baselines on three groups of evaluation metrics.

## reference
```bib
@Article{electronics12163521,
AUTHOR = {Chen, Guang and Feng, Fangxiang and Zhang, Guangwei and Li, Xiaoxu and Li, Ruifan},
TITLE = {A Visually Enhanced Neural Encoder for Synset Induction},
JOURNAL = {Electronics},
VOLUME = {12},
YEAR = {2023},
NUMBER = {16},
ARTICLE-NUMBER = {3521},
URL = {https://www.mdpi.com/2079-9292/12/16/3521},
ISSN = {2079-9292},
ABSTRACT = {The synset induction task is to automatically cluster semantically identical instances, which are often represented by texts and images. Previous works mainly consider textual parts, while ignoring the visual counterparts. However, how to effectively employ the visual information to enhance the semantic representation for the synset induction is challenging. In this paper, we propose a Visually Enhanced NeUral Encoder (i.e., VENUE) to learn a multimodal representation for the synset induction task. The key insight lies in how to construct multimodal representations through intra-modal and inter-modal interactions among images and text. Specifically, we first design the visual interaction module through the attention mechanism to capture the correlation among images. To obtain the multi-granularity textual representations, we fuse the pre-trained tags and word embeddings. Second, we design a masking module to filter out weakly relevant visual information. Third, we present a gating module to adaptively regulate the modalities&rsquo; contributions to semantics. A triplet loss is adopted to train the VENUE encoder for learning discriminative multimodal representations. Then, we perform clustering algorithms on the obtained representations to induce synsets. To verify our approach, we collect a multimodal dataset, i.e., MMAI-Synset, and conduct extensive experiments. The experimental results demonstrate that our method outperforms strong baselines on three groups of evaluation metrics.},
DOI = {10.3390/electronics12163521}
}
```
