import sys
import torch
import torch.nn as nn
from math import floor
from torch.nn import functional as F
import math
import numpy as np
import os
try:
    import triplet_loss as triplet_loss_module
except:
    import networks.triplet_loss as triplet_loss_module

def batch_cos_dist(mat):
    """计算余弦距离
    """
    above = torch.mm(mat,mat.t())
    below = torch.norm(mat,dim=1).reshape(-1,1)
    cos_sim = above/(below**2)
    cos_dist = torch.clamp(1-cos_sim,0)
    return cos_dist

def _get_anchor_positive_triplet_mask(labels,device='cpu'):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal # 忽略自相似

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def _get_triplet_mask(labels,device='cpu'):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # i !=j!= k
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    # i=j != k 
    valid_labels = ~i_equal_k & i_equal_j
    return valid_labels.to(device) & distinct_indices.to(device)

class WeightedTripletLoss(nn.Module):
    def __init__(self,use_hardest=False,weighted_factor=0.5,margin=1.0,device='cpu'):
        super().__init__()
        self.weighted_factor = weighted_factor
        self.margin=margin
        self.device=torch.device(device)
        self.use_hardest=use_hardest

    def forward(self,feats,labels):
        assert len(feats)==2
        images,tags = feats
        pairwise_dist = self.cal_distance(tags,images)
        if True:
            pairwise_dist = torch.exp(pairwise_dist)
        if self.use_hardest:
            return self.hard_triplet_loss(pairwise_dist,labels)
        else:
            triplet_loss, fraction_positive_triplets= self.all_triplet_loss(pairwise_dist,labels)
            return triplet_loss

    def cal_distance(self,tags,images):

        weighted_tags_dist = self.weighted_factor * batch_cos_dist(tags)
        weighted_images_dist = (1-self.weighted_factor) * batch_cos_dist(images)
        pairwise_dist = torch.add(weighted_tags_dist,weighted_images_dist)
        return pairwise_dist

    def hard_triplet_loss(self,pairwise_dist,labels):
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels,self.device).float()
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
        
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.margin
        tl = torch.clamp(tl,0)
        triplet_loss = tl.mean()

        return triplet_loss


    def all_triplet_loss(self,pairwise_dist,labels):
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask = _get_triplet_mask(labels,self.device)    

        triplet_loss = mask.float() * triplet_loss
        triplet_loss = torch.clamp(triplet_loss,0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / \
            (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets


class TripletLoss(nn.Module):
    def __init__(self,use_hardest=False,
                      margin=1.0,
                      device='cpu'):
        super().__init__()
        self.margin=margin
        self.device=torch.device(device)
        self.use_hardest=use_hardest
        

    def forward(self,feats,labels):
        if self.use_hardest:
            loss_values = triplet_loss_module.batch_hard_triplet_loss(labels,feats,self.margin,False, self.device,"eud")
        else:
            loss_values,_ = triplet_loss_module.batch_all_triplet_loss(labels,feats,self.margin,False,self.device)

        return loss_values

class ExpTripletLoss(nn.Module):
    def __init__(self,use_hardest=False,
                      margin=1.0,
                      device='cpu'):
        super().__init__()
        self.margin=margin
        self.device=torch.device(device)
        self.use_hardest=use_hardest
        

    def forward(self,feats,labels):
        if self.use_hardest:
            loss_values = triplet_loss_module.exp_batch_hard_triplet_loss(labels,feats,self.margin,False, self.device,"eud")
        else:
            loss_values,_ = triplet_loss_module.batch_all_triplet_loss(labels,feats,self.margin,False,self.device)

        return loss_values

if __name__ == '__main__':
    # loss_fn = WeightedTripletLoss(True)
    loss_fn = TripletLoss(use_hardest=True,margin=1.0)
    a=torch.rand(10,16)
    b=torch.rand(10,32)
    l = torch.randint(0,5,[10])
    print(l)
    loss_values = loss_fn(a,l)
    print(loss_values)
