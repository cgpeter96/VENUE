from networks.layers import (ResidulMultipleLayer,MLP,MultiLevelEmbLayer,GateComponentV2)
import torch

from networks.modules import TW_AMG

if __name__ == "__main__":

    
    backbone = TW_AMG(2048,word_vocab_size=1000, word_vocab_dim=200,
                      tag_vocab_size=1000, tag_vocab_dim=200,
                      compact_dim=512,
                      output_dim=512)
    img =torch.rand(1,16,2048)
    words = torch.randint(0,1000,[1,7])

    tag = torch.randint(0,1000,[1])
    feat = backbone(img,words,tag)
    print(feat)
    if len(feat)==1:
        print(feat[0].shape)
    else:
        print(feat[0].shape)
        print(feat[1].shape)
    
    print("hello world")