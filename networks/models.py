import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from modules  import TrimSynBackbone,VisualBackbone,TextBackbone,TW_AMG
except:
    from networks.modules  import (TrimSynBackbone,
        TrimSynBackboneWithoutTM,VisualBackbone,
        TextBackbone,TrimSynBackboneWithoutNLA,
        TrimSynBackboneWithoutGate,TrimSynBackboneWithoutMasking,
        AsyemmetricBackbone,TW_AMG)
class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def save_model(self, model_filename):
        state_dict = {
            "model":self.state_dict()
        }
        model_dir = os.path.split(model_filename)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(state_dict, model_filename)


    def load_model(self, model_filename):
        print('loading model from ', model_filename)
        state_dict = torch.load(model_filename, map_location='cpu')
        self.load_state_dict(state_dict["model"])

class TrimSyn(Base):
    def __init__(self,img_dim,
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embedding=None,
                      model_type="full"):
        super().__init__()
        self.img_dim = img_dim
        self.txt_vocab_size = txt_vocab_size
        self.txt_dim = txt_dim
        self.compact_dim = compact_dim
        self.output_dim = output_dim
        self.op = op
        self.word_embedding = word_embedding
        self.backbone = self.build_backbone(model_type=model_type)

    def build_backbone(self,model_type):
        if model_type=="full":
            return TrimSynBackbone(self.img_dim,
                                    self.txt_vocab_size,
                                    self.txt_dim,
                                    self.compact_dim,
                                    self.output_dim,
                                    self.op,
                                    self.word_embedding)
        elif model_type=="withoutTM":
            return TrimSynBackboneWithoutTM(self.img_dim,
                                        self.txt_vocab_size,
                                        self.txt_dim,
                                        self.compact_dim,
                                        self.output_dim,
                                        self.op,
                                        self.word_embedding)
        elif model_type=="withoutNLA":
            return TrimSynBackboneWithoutTM(self.img_dim,
                                        self.txt_vocab_size,
                                        self.txt_dim,
                                        self.compact_dim,
                                        self.output_dim,
                                        self.op,
                                        self.word_embedding)
        elif model_type=="withoutGate":
            return TrimSynBackboneWithoutGate(self.img_dim,
                                        self.txt_vocab_size,
                                        self.txt_dim,
                                        self.compact_dim,
                                        self.output_dim,
                                        self.op,
                                        self.word_embedding)
        elif model_type=="withoutMasking":
            return TrimSynBackboneWithoutMasking(self.img_dim,
                                        self.txt_vocab_size,
                                        self.txt_dim,
                                        self.compact_dim,
                                        self.output_dim,
                                        self.op,
                                        self.word_embedding)
        elif model_type=="asym":
            return AsyemmetricBackbone(self.img_dim,
                                        self.txt_vocab_size,
                                        self.txt_dim,
                                        self.compact_dim,
                                        self.output_dim,
                                        self.op,
                                        self.word_embedding)
        else:
            raise Exception("model_type error:{}".format(model_type))

    def forward(self, query_text, pos_text, neg_text, query_image_set, pos_image_set, neg_image_set):
        """
        Args:
            query_text:B x 100                  -> B X D
            query_image_set: B x 50 x2048       -> B X D 
        """
        query_feat = self.forward_instance(query_text, query_image_set)
        pos_feat = self.forward_instance(pos_text, pos_image_set)
        neg_feat = self.forward_instance(neg_text, neg_image_set)
        return query_feat, pos_feat, neg_feat

    def forward_instance(self,text,images):
        """
        Args:
            text(B):
            images(Bx50x2048)
        """
        gate_feat = self.backbone(images,text)
        return gate_feat


class SingleSyn(Base):
    def __init__(self,img_dim,
                      txt_vocab_size,
                      txt_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embedding=None,
                      model_type="text"):
        super().__init__()
        self.img_dim = img_dim
        self.txt_vocab_size = txt_vocab_size
        self.txt_dim = txt_dim
        self.compact_dim = compact_dim
        self.output_dim = output_dim
        self.op = op
        self.word_embedding = word_embedding
        self.backbone = self.build_backbone(model_type=model_type)

    def build_backbone(self,model_type):
        if model_type=="text":
            return TextBackbone(self.img_dim,
                            self.txt_vocab_size,
                            self.txt_dim,
                            self.compact_dim,
                            self.output_dim,
                            self.op,
                            self.word_embedding)
        elif model_type=="visual":
            return VisualBackbone(self.img_dim,
                            self.txt_vocab_size,
                            self.txt_dim,
                            self.compact_dim,
                            self.output_dim,
                            self.op,
                            self.word_embedding)



    def forward(self, query_text, pos_text, neg_text, query_image_set, pos_image_set, neg_image_set):
        """
        Args:
            query_text:B x 100                  -> B X D
            query_image_set: B x 50 x2048       -> B X D 
        """
        query_feat = self.forward_instance(query_text, query_image_set)
        pos_feat = self.forward_instance(pos_text, pos_image_set)
        neg_feat = self.forward_instance(neg_text, neg_image_set)
        return query_feat, pos_feat, neg_feat

    def forward_instance(self,feat):
        """
        Args:
            text(B):
            images(Bx50x2048)
        """
        out_feat = self.backbone(feat)
        return out_feat

class NewModel(Base):
    def __init__(self,img_dim,
                      tag_vocab_size, 
                      tag_vocab_dim,
                      compact_dim,
                      output_dim,
                      op="concat",
                      word_embedding=None,
                      model_type="model_1",
                      word_vocab_size=1000, 
                      word_vocab_dim=200,
                      tag_embedding=None,
                      emb_pool_type="lse",
                      emb_param_r=6):
        super().__init__()
        self.img_dim = img_dim
        self.tag_vocab_size = tag_vocab_size
        self.tag_vocab_dim = tag_vocab_dim
        self.compact_dim = compact_dim
        self.output_dim = output_dim
        self.op = op
        self.word_embedding = word_embedding
        self.word_vocab_size=word_vocab_size
        self.word_vocab_dim=word_vocab_dim
        self.tag_embedding=tag_embedding
        self.emb_pool_type=emb_pool_type
        self.emb_param_r=emb_param_r
        self.backbone = self.build_backbone(model_type=model_type)

    def build_backbone(self,model_type):
        if model_type=="model_1":
            return TW_AMG(img_dim=self.img_dim,
                      word_vocab_size=self.word_vocab_size, word_vocab_dim=self.word_vocab_dim,
                      tag_vocab_size=self.tag_vocab_size, tag_vocab_dim=self.tag_vocab_dim,
                      compact_dim=self.compact_dim,
                      output_dim=self.output_dim,
                      op=self.op,
                      word_embeding=self.word_embedding,
                      tag_embedding=self.tag_embedding,
                      emb_pool_type=self.emb_pool_type,
                      emb_param_r=self.emb_param_r
                    )
        else:
            return None

    def forward(self, query_tag, pos_tag, neg_tag, 
                      query_image_set, pos_image_set, neg_image_set,
                      query_words,pos_words, neg_words):
        """
        Args:
            query_tag:B x 1
            query_image_set: B x N x 2048
            query_words:BxN
        """
        query_feat = self.forward_instance( query_image_set,query_words,query_tag)
        pos_feat = self.forward_instance( pos_image_set,pos_words,pos_tag)
        neg_feat = self.forward_instance( neg_image_set,neg_words,neg_tag)
        return query_feat, pos_feat, neg_feat

    def forward_instance(self,images,words,*args):
        """
        Args:
            text(B):
            images(Bx50x2048)
        """
        out_feat = self.backbone(images,words,args[0])
        return out_feat

if __name__ == '__main__':
    # model = SingleSyn(2048,1000,50,1024,512,op="add",word_embedding=None,model_type="visual")
    model = NewModel(img_dim=2048,
                      tag_vocab_size=1000, tag_vocab_dim=200,
                      compact_dim=512,
                      output_dim=512,
                      op="concat",
                      word_embedding=None,
                      model_type="model_1",
                      word_vocab_size=1000, word_vocab_dim=200,
                      tag_embedding=None)
    img = torch.rand(10,15,2048)
    words = torch.randint(0,1000,[10,7])
    tag = torch.randint(0,1000,[10])
    gf = model.forward_instance(img,words,tag)
    print(gf.shape)