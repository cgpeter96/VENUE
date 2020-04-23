"""
backbone & module
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
try:
    from layers import (init_linear, GateComponent,AvgPoolNet,NONLocalBlock1D)
except:
    from networks.layers import (init_linear, GateComponent,AvgPoolNet,NONLocalBlock1D)


class NaiveVisualEncoder(nn.Module):
    def __init__(self,images_feat_dim=2048,output_dim=512):
        super().__init__()
        self.image_transform =nn.Sequential(
                                nn.Linear(images_feat_dim,output_dim),
                                nn.ReLU()
                              ) 

    def forward(self, images_feat):
        """
        Args:
            images_feat(Bx N x D ): 图像集特征
        """
        transform_feat = self.image_transform(images_feat) # Bx50x200
        return transform_feat 

class DeepVisualEncoder(nn.Module):
    def __init__(self,images_feat_dim=2048,output_dim=512):
        super().__init__()
        self.image_transform =nn.Sequential(
                                nn.Linear(images_feat_dim,images_feat//2),
                                nn.ReLU(),
                                nn.Linear(images_feat_dim//2,output_dim),
                                nn.ReLU(),
                                nn.Linear(output_dim,output_dim),
                                nn.ReLU(),
                              ) 

    def forward(self, images_feat):
        """
        Args:
            images_feat(Bx N x D ): 图像集特征
        """
        transform_feat = self.image_transform(images_feat) # Bx50x200
        return transform_feat 

class NonLocalVisualEncoder(nn.Module):
    def __init__(self,images_feat_dim=2048,output_dim=512):
        super().__init__()
        self.attention_net = NONLocalBlock1D(images_feat_dim, inter_channels=None, sub_sample=True, bn_layer=True)
        # self.image_transform =nn.Sequential(
        #                         nn.Linear(images_feat_dim,output_dim),
        #                         nn.Tanh()
        #                       ) 
        self.image_transform = lambda x:x

    def forward(self,images_feat):
        """
        Args:
            images_feat(Bx N x D ): 图像集特征
        """
        images_feat_ = images_feat.permute(0,2,1).contiguous() # B x 2048 x 50
        images_attention_feat,attenion_weight = self.attention_net(images_feat_)
        images_attention_feat = images_attention_feat.permute(0,2,1).contiguous() # B x 50 x2048
        transform_feat = self.image_transform(images_attention_feat) # Bx50x200
        return transform_feat 


class BaseTextEncoder(nn.Module):
    """TextEncoder 
    """
    def __init__(self):
        super().__init__()
        self.emb = None

    def get_word_emb(self, label):
        """
        Args:
            label(B):是一维的表示
        """
        return self.emb(label)

    def load_pretrain_embedding(self, pretrained_emb, embed_fine_tune=False):
        if pretrained_emb != None:
            # 读取word embedding
            
            pretrained_embedding = pretrained_emb.vectors
            padding_embedding = np.zeros([1, pretrained_embedding.shape[1]])
            pretrained_embedding = np.row_stack(
                [padding_embedding, pretrained_embedding])
            self.emb.weight.data.copy_(
                torch.from_numpy(pretrained_embedding))
            self.emb.weight.requires_grad = embed_fine_tune

class NaiveTextEncoder(BaseTextEncoder):
    def __init__(self, vocab_size, vocab_dim, out_emb_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_dim)
        # self.linear = nn.Sequential(
        #     nn.Linear(vocab_dim, out_emb_size),
        #     nn.Tanh(),
        # )
        # init_linear(self.linear)

    def forward(self, label):
        """
        Args:
            label(B):是一维的表示
        """
        label_emb = self.emb(label)
        return label_emb

class DeepTextEncoder(BaseTextEncoder):
    def __init__(self, vocab_size, vocab_dim, out_emb_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_dim)
        self.linear = nn.Sequential(
            nn.Linear(vocab_dim, vocab_dim**2),
            nn.ReLU(),
            nn.Linear(vocab_dim**2, out_emb_size),
            nn.ReLU(),
            nn.Linear(out_emb_size, out_emb_size),
            nn.ReLU(),
        )
        init_linear(self.linear)

    def forward(self, label):
        """
        Args:
            label(B):是一维的表示
        """
        label_emb = self.emb(label)
        return self.linear(label_emb),label_emb


# ================masking====================

class ImageTransModule(nn.Module):
    def __init__(self,img_dim, compact_dim,output_dim,**kargs):
        super().__init__()
        self.compact_image = nn.Sequential(
                                nn.Linear(img_dim,compact_dim),
                                nn.ReLU(),
                                nn.Linear(compact_dim,output_dim),
                                nn.ReLU(),
                                )
       

        self.pool =  AvgPoolNet(output_dim,output_dim)

    def forward(self,images_feat, *args):
        img_c =self.compact_image(images_feat) # Bx50x200
        return self.pool(img_c)

class ImageMaskingModule(nn.Module):
    def __init__(self,img_dim, text_dim, compact_dim,output_dim):
        super().__init__()
        self.compact_image = nn.Sequential(
                                nn.Linear(img_dim,img_dim//2),
                                nn.ReLU(),
                                nn.Linear(img_dim//2,compact_dim),
                                nn.ReLU())
        self.compact_text = nn.Sequential(
                                nn.Linear(text_dim,compact_dim),
                                nn.Tanh(),
                                nn.Linear(compact_dim,compact_dim),
                                nn.Tanh())
        self.img_linear = nn.Linear(img_dim,compact_dim)

        self.pool =  AvgPoolNet(compact_dim,output_dim)

    def forward(self,images_feat, text_feat):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B X D):
        """
        img_c =self.compact_image(images_feat) # Bx50x200
        txt_c = self.compact_text(text_feat).unsqueeze(dim=2) # B x 200x 1
        img_out = self.img_linear(images_feat)
        mask_vector = torch.sigmoid(torch.bmm(img_c,txt_c)) # B x 50x1
        
        mask_img_feat = torch.mul(mask_vector,img_out)
        global_mask_img_feat = self.pool(mask_img_feat)
        return global_mask_img_feat

class RawImageMaskingModule(nn.Module):
    def __init__(self,img_dim, text_dim, compact_dim,output_dim):
        super().__init__()

        
        self.compact_image = nn.Sequential(
                                nn.Linear(img_dim,compact_dim),
                                nn.ReLU(),
                                )
        self.compact_text = nn.Sequential(
                                nn.Linear(text_dim,compact_dim),
                                nn.Tanh(),
                                )
        self.img_linear = nn.Linear(img_dim,compact_dim)

        self.pool =  AvgPoolNet(img_dim,output_dim)
        self.mask_vec = None

    def forward(self,images_feat, text_feat):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B X D):
        """

        img_c =self.compact_image(images_feat) # Bx50x200
        txt_c = self.compact_text(text_feat).unsqueeze(dim=2) # B x 200x 1
        
        mask_vector = torch.sigmoid(torch.bmm(img_c,txt_c)) # B x 50x1
        # img_out = self.img_linear(images_feat)
        self.mask_vec = mask_vector.detach().cpu().numpy()
        mask_img_feat = torch.mul(mask_vector,images_feat)

        global_mask_img_feat = self.pool(mask_img_feat)
        return global_mask_img_feat

class NonOpModule(nn.Module):
    def __init__(self,text_dim, compact_dim,output_dim,**kargs):
        super().__init__()
       

    def forward(self,text_feat, *args):
        return text_feat


class TextTransModule(nn.Module):
    def __init__(self,text_dim, compact_dim,output_dim,**kargs):
        super().__init__()
        self.compact_text = nn.Sequential(
                                nn.Linear(text_dim,compact_dim),
                                nn.Tanh(),
                                nn.Linear(compact_dim,output_dim), 
                                nn.Tanh(),
                                )

       

    def forward(self,text_feat, *args):
        return self.compact_text(text_feat) 

class TextMaskingModule(nn.Module):

    def __init__(self,img_dim, text_dim, compact_dim,output_dim):
        super().__init__()
        # self.pool =  AvgPoolNet(img_dim,img_dim)
        self.compact_image = nn.Sequential(
                                nn.Linear(img_dim,compact_dim),
                                nn.Tanh(),
                                nn.Linear(compact_dim,compact_dim),
                                nn.Tanh())
        self.compact_text = nn.Sequential(
                                nn.Linear(text_dim,compact_dim),
                                nn.Tanh(),
                                nn.Linear(compact_dim,compact_dim),
                                nn.Tanh())
        self.text_linear = nn.Linear(img_dim,compact_dim)
        self.out_linear = nn.Linear(compact_dim,output_dim)


    def forward(self,images_feat,text_feat):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B X D):
        """
        images_feat_pool = images_feat.mean(dim=1)
        img_c =self.compact_image(images_feat_pool) # Bx 200
        txt_c = self.compact_text(text_feat) # B x 200
        text_out = self.text_linear(text_feat)
        mask_vector = torch.sigmoid(torch.mul(txt_c,img_c))

        return self.out_linear(mask_vector * text_out)

# ==================backbone=======================
class TrimSynBackbone(nn.Module):
    def __init__(self,img_dim,
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embeding=None):
        """
        op:[add|concat|none]
        """
        super().__init__()
        self.visual_enc = NonLocalVisualEncoder(img_dim,compact_dim)
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, compact_dim)
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        self.img_mask = ImageMaskingModule(img_dim, compact_dim, compact_dim,output_dim)
        self.txt_mask = TextMaskingModule(compact_dim, compact_dim, compact_dim,output_dim)
        self.op = op
        self.gate = GateComponent(output_dim,output_dim,output_dim,output_dim,op=self.op)


    def forward(self, images, text):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """

        images_feat = self.visual_enc(images)
        text_feat = self.txt_enc(text)

        mask_img_feat  = self.img_mask(images_feat,text_feat)
        mask_txt_feat  = self.txt_mask(images_feat,text_feat)

        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat

class TrimSynBackboneWithoutTM(nn.Module):
    def __init__(self,img_dim,
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embeding=None):
        """
        op:[add|concat|none]
        img_dim = 2048
        compact_dim = 200
        output_dim = 512
        """
        super().__init__()
        self.visual_enc = NonLocalVisualEncoder(img_dim,compact_dim)
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, output_dim)
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        self.img_mask = RawImageMaskingModule(img_dim, txt_dim, compact_dim,output_dim)
        self.txt_mask = NonOpModule(compact_dim, compact_dim, output_dim)
        self.op = op
        self.gate = GateComponent(output_dim,output_dim,output_dim,output_dim,op=self.op)

    def forward(self, images, text):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """
        images_feat = self.visual_enc(images)
        # 注意这部分
        text_feat,word_emb = self.txt_enc(text)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        mask_txt_feat  = self.txt_mask(text_feat)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat


if __name__ == '__main__':
    backbone = TrimSynBackbone(2048,1000,50,1024,512,op="concat")
    img =torch.rand(1,16,2048)
    txt = torch.randint(0,1000,[1])
    feat = backbone(img,txt)
    if len(feat)==1:
        print(feat[0].shape)
    else:
        print(feat[0].shape)
        print(feat[1].shape)


