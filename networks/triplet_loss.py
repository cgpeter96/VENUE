#!/usr/bin/env python3
# -*- coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, device='cpu',dist_type='eud'):
    """Build the triplet loss over a batch of embeddings.(在一个batch 中labels 应该是有相同的 )

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    if dist_type=='eud':

        pairwise_dist = _pairwise_distances(embeddings, squared=squared)
        # print(pairwise_dist.shape)
    elif dist_type=='kl':
        pairwise_dist = _KL_distance(embeddings)
        # print(pairwise_dist.shape)qq
    else:
        raise Exception("dist type error")
    # print(pairwise_dist)
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(
        labels, device).float()

    # print(mask_anchor_positive)
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
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
    tl = hardest_positive_dist - hardest_negative_dist + margin
    # print("NEG:",hardest_negative_dist)
    # print("POS",hardest_positive_dist)
    # tl[tl < 0] = 0
    tl = torch.clamp(tl,0)
    triplet_loss = tl.mean()

    return triplet_loss

def batch_hard_multi_loss(labels, embeddings, margin, squared=False, device='cpu',dist_type='eud',alpha=0.5):
    """Build the triplet loss over a batch of embeddings.(在一个batch 中labels 应该是有相同的 )

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    if dist_type=='eud':

        pairwise_dist = _pairwise_distances(embeddings, squared=squared)
        # print(pairwise_dist.shape)
    elif dist_type=='kl':
        pairwise_dist = _KL_distance(embeddings)
        # print(pairwise_dist.shape)qq
    else:
        raise Exception("dist type error")
    # print(pairwise_dist)
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(
        labels, device).float()

    # print(mask_anchor_positive)
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
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
    tl = hardest_positive_dist - hardest_negative_dist + margin
    # print("NEG:",hardest_negative_dist)
    # print("POS",hardest_positive_dist)
    # tl[tl < 0] = 0
    tl = torch.clamp(tl,0)
    triplet_loss = tl.mean()

    # cos_matrix = []
    # for i in embeddings:
    #     r = torch.cosine_similarity(i.repeat(embeddings.size(0),1),embeddings)
    #     cos_matrix.append(r)
    # cos_matrix = torch.cat(cos_matrix,dim=0)


    dist = torch.sigmoid(1-pairwise_dist.reshape(-1))

    cos_label = []
    for i in labels:
        for j in labels:
            cos_label.append(i==j)
    cos_label = torch.stack(cos_label,dim=0).float()
    bcs_loss = F.binary_cross_entropy(dist,cos_label)



    return triplet_loss+0.5*bcs_loss

def batch_all_triplet_loss(labels, embeddings, margin, squared=False,device="cpu"):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels,device)

    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    # triplet_loss[triplet_loss < 0] = 0
    triplet_loss = torch.clamp(triplet_loss,0)
    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / \
        (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(
        0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances



def _get_triplet_mask(labels,device):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels.to(device) & distinct_indices.to(device)


def _get_anchor_positive_triplet_mask(labels, device='cpu'):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

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


def cal_kl_dist(predict,target,eps = 1e-7):
    target = target.detach()

    # KL(T||I) = \sum T(logT-logI)
    predict_ = predict+eps
    target_ = target+eps
    logI = predict_.log()
    logT = target_.log()
    TlogTdI = target * (logT - logI)
    kld = TlogTdI.sum(1)
    return kld

def _KL_distance(embeddings):
    # kl散度不会小于0
    pairs = []
    for emb in embeddings:
        emb_r = emb.repeat(embeddings.size(0),1)
        dist = cal_kl_dist(emb_r,embeddings)+cal_kl_dist(embeddings,emb_r)
        pairs.append(dist)
    pairs_dist = torch.stack(pairs)
    return pairs_dist


class KLCompareLoss(nn.Module):
    def __init__(self,margin1,margin2):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        

    def forward(self,query,pos,neg):
        pos_dist = cal_kl_dist(query,pos)+cal_kl_dist(pos,query)

        neg_dist = torch.clamp(self.margin2-cal_kl_dist(query,neg),0)+torch.clamp(self.margin2-cal_kl_dist(neg,query),0)
        return (pos_dist+neg_dist).mean(0)