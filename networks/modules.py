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
    from layers import (ResidulMultipleLayer,MLP,MultiLevelEmbLayer,GateComponentV2)
except:
    from networks.layers import (init_linear, GateComponent,AvgPoolNet,NONLocalBlock1D)
    from networks.layers import (ResidulMultipleLayer,MLP,MultiLevelEmbLayer,GateComponentV2)


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
                                nn.Linear(images_feat_dim,images_feat_dim//2),
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
        self.attention_net = NONLocalBlock1D(images_feat_dim, inter_channels=None, sub_sample=False, bn_layer=False)
        # self.image_transform =nn.Sequential(
        #                         nn.Linear(images_feat_dim,output_dim),
        #                         nn.ReLU()
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
    def __init__(self, vocab_size, vocab_dim, *args,**kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_dim)

    def forward(self, text):
        """
        Args:
            text(B):是一维的表示
        """
        return self.emb(text)
        

class DeepTextEncoder(BaseTextEncoder):
    def __init__(self, vocab_size, vocab_dim, out_emb_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_dim)
        self.linear = nn.Sequential(
            nn.Linear(vocab_dim, vocab_dim**2),
            nn.Tanh(),
            nn.Linear(vocab_dim**2, out_emb_size),
            nn.Tanh(),
            nn.Linear(out_emb_size, out_emb_size),
            nn.Tanh(),
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
                                nn.ReLU(), # raw is tanh
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
        # print(img_c.shape)
        txt_c = self.compact_text(text_feat).unsqueeze(dim=2) # B x 200x 1
        # print(txt_c.shape)
        mask_vector = torch.sigmoid(torch.bmm(img_c,txt_c)) # B x 50x1
        # print(mask_vector.shape)
        # img_out = self.img_linear(images_feat)
        self.mask_vec = mask_vector.detach().cpu().numpy()
        mask_img_feat = torch.mul(mask_vector,images_feat)
        # print(mask_img_feat.shape)
        global_mask_img_feat = self.pool(mask_img_feat)
        # print(global_mask_img_feat.shape)
        return global_mask_img_feat


class NonRawImageMaskingModule(nn.Module):
    """
    不产生mask vector
    """
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

        global_mask_img_feat = self.pool(images_feat)
        return global_mask_img_feat

class NonOpModule(nn.Module):
    def __init__(self,text_dim, compact_dim,output_dim,**kargs):
        super().__init__()
       

    def forward(self,text_feat, *args):
        return text_feat

class LinearOpModule(nn.Module):
    def __init__(self,text_dim, compact_dim,output_dim,**kargs):
        super().__init__()
        self.linear = nn.Linear(text_dim,output_dim)

    def forward(self,text_feat, *args):
        return self.linear(text_feat)


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
        self.text_linear = nn.Linear(compact_dim,compact_dim)
        self.out_linear = nn.Linear(compact_dim,output_dim)
        self.mask_vec = None
        self.pre = None


    def forward(self,images_feat,text_feat):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B X D):
        """
        images_feat_pool = images_feat.mean(dim=1)
        # print(images_feat_pool.shape)
        img_c =self.compact_image(images_feat_pool) # Bx 200
        # print(img_c.shape)
        txt_c = self.compact_text(text_feat) # B x 200
        # print(txt_c.shape)
        text_out = self.text_linear(text_feat)
        # print(text_out.shape)
        mask_vector = torch.sigmoid(torch.mul(txt_c,img_c))
        # self.mask_vec = mask_vector.detach().cpu().numpy()
        # print(mask_vector.shape)
        out = self.out_linear(mask_vector * text_out)
        self.mask_vec = (text_feat.detach().cpu().numpy(),out.detach().cpu().numpy())
        return out

class NonGate(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()

    def forward(self, img, word):
        return img,word


# ==================backbone=======================

class TextBackbone(nn.Module):
    def __init__(self,img_dim, #用不着
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,  
                      op="concat", #用不着
                      word_embeding=None):
        super().__init__()
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, compact_dim)
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        self.out_linear = nn.Linear(compact_dim,output_dim)

    def forward(self, text):
        text_feat,raw_feat = self.txt_enc(text)
        output_feat = self.out_linear(text_feat)
        return output_feat

class VisualBackbone(nn.Module):
    def __init__(self,img_dim, 
                      txt_vocab_size,#用不着
                      txt_dim,#用不着
                      compact_dim,
                      output_dim,  
                      op="concat", #用不着
                      word_embeding=None):
        super().__init__()
        self.visual_enc = DeepVisualEncoder(images_feat_dim=img_dim,output_dim=compact_dim)
        
        self.out_linear = nn.Linear(compact_dim,output_dim)

    def forward(self,image):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """
        
        B,N,D = image.size(0),image.size(1),image.size(2)
        image = image.reshape(B,N,D)
        image = image.mean(dim=1)
        visual_feat =  self.visual_enc(image)
        output_feat = self.out_linear(visual_feat)

        return output_feat



class TrimSynBackbone(nn.Module):
    def __init__(self,img_dim,
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,
                      op="none",
                      word_embeding=None):
        """
        op:[add|concat|none]
        """
        super().__init__()
        # 用过ml
        self.visual_enc = NonLocalVisualEncoder(img_dim,compact_dim)
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, compact_dim)
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        # self.img_mask = ImageMaskingModule(img_dim, compact_dim, compact_dim,output_dim)
        self.img_mask = RawImageMaskingModule(img_dim, compact_dim, compact_dim,output_dim)
        self.txt_mask = TextMaskingModule(img_dim, compact_dim, compact_dim,output_dim)
        self.op = op
        self.gate = GateComponent(output_dim,output_dim,output_dim,output_dim,op=self.op)


    def forward(self, images, text):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """
        # print(images.shape)
        images_feat = self.visual_enc(images)
        # print(images_feat.shape)
        # print(text.shape)
        text_feat,word_emb = self.txt_enc(text)
        # print(text_feat.shape)
        mask_img_feat  = self.img_mask(images_feat,text_feat)
        # print("a",mask_img_feat.shape)
        mask_txt_feat  = self.txt_mask(images_feat,text_feat)
        # print("b",mask_txt_feat.shape)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)

        return gate_feat

class TrimSynBackboneWithoutGate(nn.Module):
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
        self.gate = NonGate(output_dim,output_dim,output_dim,output_dim,op=self.op)

    def forward(self, images, text):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """
        images_feat = self.visual_enc(images)
        # 注意这部分
        # print(images_feat.shape)
        text_feat,word_emb = self.txt_enc(text)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        mask_txt_feat  = self.txt_mask(text_feat)
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
        # print(images_feat.shape)
        text_feat,word_emb = self.txt_enc(text)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        mask_txt_feat  = self.txt_mask(text_feat)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat

class TrimSynBackboneWithoutMasking(nn.Module):
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
        self.img_mask = NonRawImageMaskingModule(img_dim, txt_dim, compact_dim,output_dim)
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
        # print(images_feat.shape)
        text_feat,word_emb = self.txt_enc(text)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        mask_txt_feat  = self.txt_mask(text_feat)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat


class TrimSynBackboneWithoutNLA(nn.Module):
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
        self.visual_enc = DeepVisualEncoder(img_dim,compact_dim)
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, output_dim)
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        # self.img_mask = NonRawImageMaskingModule(img_dim, txt_dim, compact_dim,output_dim)
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
        B,N,D = images.size(0),images.size(1),images.size(2)
        images = images.reshape(B,N,D)
        images_feat = self.visual_enc(images)
        # 注意这部分
        # print(images_feat.shape)
        text_feat,word_emb = self.txt_enc(text)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        mask_txt_feat  = self.txt_mask(text_feat)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat


class AsyemmetricBackbone(nn.Module):
    """

    """
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
        self.visual_enc = NonLocalVisualEncoder(img_dim,compact_dim) # 
        # self.visual_enc = DeepVisualEncoder(img_dim,img_dim)
        self.txt_enc = DeepTextEncoder(txt_vocab_size, txt_dim, output_dim) # or NaiveTextEncoder
        if word_embeding is not None:
            logger.info("load pretrain embedding")
            self.txt_enc.load_pretrain_embedding(word_embeding)
        self.img_mask = RawImageMaskingModule(img_dim, txt_dim, compact_dim,output_dim)
        self.txt_mask = LinearOpModule(output_dim, compact_dim, output_dim)
        self.op = op
        self.gate = GateComponent(output_dim,output_dim,output_dim,output_dim,op=self.op)
        # self.gate = NonGate(output_dim,output_dim,output_dim,output_dim,op=self.op)

    def forward(self, images, text):
        """
        Args:
            images_feat(B x N X D):
            text_feat(B ):
        """
        B,N,D = images.size(0),images.size(1),images.size(2)
        images = images.reshape(B,N,D)
        images_feat = self.visual_enc(images)

        # 注意这部分
        # print(images_feat.shape)
        text_feat,word_emb = self.txt_enc(text)
        # print(word_emb.shape)
        mask_img_feat  = self.img_mask(images_feat,word_emb)
        # print(mask_img_feat.shape)
        # print("--",text_feat.shape)
        mask_txt_feat  = self.txt_mask(text_feat)
        
        # print(mask_txt_feat.shape)
        gate_feat = self.gate(mask_img_feat,mask_txt_feat)
        return gate_feat

# ==================================
class TW_AMG(nn.Module):
    def __init__(self,img_dim,
                      word_vocab_size, word_vocab_dim,
                      tag_vocab_size, tag_vocab_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embeding=None,
                      tag_embedding=None,
                      emb_pool_type="lse",
                      emb_param_r=6):
        """
        op:[add|concat|none]
        img_dim = 2048
        compact_dim = 200
        output_dim = 512
        """
        super().__init__()
        # super().__init__()
        self.visual_enc = NonLocalVisualEncoder(img_dim,compact_dim) # 
        self.mg_text_enc = MultiLevelEmbLayer(word_embeding,
                                            word_vocab_size, word_vocab_dim,
                                            tag_embedding,
                                            tag_vocab_size, tag_vocab_dim,
                                            pool_type=emb_pool_type,param_r=emb_param_r) 
        # TODO mg后要不要nn
        self.visual_mask = RawImageMaskingModule(img_dim,
                                            word_vocab_dim+tag_vocab_dim,
                                            compact_dim,output_dim)
        self.txt_prj_layer = MLP([word_vocab_dim+tag_vocab_dim,512,output_dim])
        self.gate = GateComponent(output_dim,output_dim,output_dim,output_dim,op=op)
        # TODO Gate后要不要接nn


    def forward(self,images,words,*args):
        """
        Args:
            images_feat:(B x N x D)
            words:(B x N)
            args[0]:(B)
        """

        tag=args[0]
        B,N,D = images.size(0),images.size(1),images.size(2)
        images = images.reshape(B,N,D)
        #rec
        images_feat = self.visual_enc(images)
        mg_text_feat = self.mg_text_enc(words,tag)
        #mask
        mask_visual_feat = self.visual_mask(images_feat,mg_text_feat)
        #gate
        pjt_text_feat = self.txt_prj_layer(mg_text_feat)
        gate_feat = self.gate(mask_visual_feat,pjt_text_feat)
        return gate_feat


if __name__ == '__main__':
    # backbone = TrimSynBackboneWithoutMasking(2048,1000,50,1024,512,op="concat")
    # backbone = AsyemmetricBackbone(2048,1000,50,1024,512,op="concat")

    backbone = TW_AMG(2048,word_vocab_size=1000, word_vocab_dim=200,
                      tag_vocab_size=1000, tag_vocab_dim=200,
                      compact_dim=512,
                      output_dim=512,)
    img =torch.rand(1,16,2048)
    words = torch.randint(0,1000,[1,7])
    tag = torch.randint(0,1000,[1])
    feat = backbone(img,word,tag)
    if len(feat)==1:
        print(feat[0].shape)
    else:
        print(feat[0].shape)
        print(feat[1].shape)


