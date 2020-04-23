"""
存放基础层
"""

import torch
import torch.nn as nn
from math import floor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

def check_tensor(tensor):
    if torch.cuda.is_available():
        return tensor.to("cuda")
    return tensor


def init_linear(net, type='xavier_normal'):
    init_op = {
        'xavier_normal': nn.init.xavier_normal_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,

    }

    for l in net:
        if isinstance(l, nn.Linear):
            init_op[type](l.weight)



class FC(nn.Module):
    def __init__(self):
        super().__init__()

    def initilize(self):
        """
        initilize linear layer
        """
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

class GateComponent(FC):
    """
    balance the importance image and text 
    """

    def __init__(self, img_dim, word_dim, out_img_dim,out_word_dim, op='add'):
        """
        if op == concat:
            output feature shape = out_img_dim +out_word_dim
        """
        super().__init__()
        self.wv = nn.Linear(img_dim, out_img_dim)
        self.wt = nn.Linear(word_dim, out_word_dim)
        self.wv_hat = nn.Linear(out_img_dim, out_img_dim)  # 应可以用同一层
        self.wt_hat = nn.Linear(out_word_dim, out_word_dim)
        self.op = op
        self.initilize()

    def forward(self, img, word):
        '''
        
        Args:
            img(BXD):图像表示
            word(BXD):文本表示
        '''
    
        v = self.wv(img)
        t = self.wt(word)
        v_hat = self.wv_hat(v)
        t_hat = self.wt_hat(t)
        g = torch.sigmoid(v_hat + t_hat)
        if self.op == 'add':
            return torch.add(torch.mul(g, v), torch.mul((1 - g), t))
        elif self.op =="none":
            return torch.mul(g, v), torch.mul((1 - g), t)
        elif self.op =="concat":
            return torch.cat([torch.mul(g, v), torch.mul((1 - g), t)],dim=1)
        else:
            raise Exception("No op:{}".format(self.op))


class AvgPoolNet(nn.Module):
    """
    将16x50x2048 -> 聚合为16x512
    """
    def __init__(self,input_dim=2048,output_dim=512):
        super().__init__()
        self.trans=nn.Linear(input_dim,output_dim)

    def forward(self,feat):
        trans_feat = self.trans(feat)
        return torch.mean(trans_feat,dim=1)


#======================Non local attention=======================================

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        self.attn_weight = None

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # print(x.shape)
        # (b,c/2,t,h,w)->(b,c/2,t*h*w)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (b,t*h*w,c)
        # print(g_x.shape)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # print(theta_x.shape)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # print(phi_x.shape)
        f = torch.matmul(theta_x, phi_x)  # (b,t*h*w,t*h*w)
        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)  # (b,t*h*w,t*h*w)
        # self.attn_weight = f_div_C.detach().cpu().numpy()
        y = torch.matmul(f_div_C, g_x)  # (b,t*h*w,c)
        y = y.permute(0, 2, 1).contiguous()  # (b,c,t*h*w)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z, f_div_C

class NONLocalBlock1D(_NonLocalBlockND):
    """处理16 x 2048 x50的数据
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class NONLocalBlock3D(_NonLocalBlockND):
    """处理 16x 2048 x50 x7 x7的数据
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    main()