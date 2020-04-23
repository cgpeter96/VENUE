import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from modules  import TrimSynBackbone
except:
    from networks.modules  import TrimSynBackbone,TrimSynBackboneWithoutTM
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

    def forward_instance(self,text,images,):
        """
        Args:
            text(B):
            images(Bx50x2048)
        """
        gate_feat = self.backbone(images,text)
        return gate_feat



if __name__ == '__main__':
    model = TrimSyn(2048,1000,50,1024,512,op="add",word_embedding=None)
    img = torch.rand(10,15,2048)
    txt = torch.randint(0,1000,[10])
    gf = model.forward_instance(txt,img)
    print(gf[0].shape)