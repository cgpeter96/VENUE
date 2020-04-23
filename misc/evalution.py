import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
import itertools
import networkx as nx
from tqdm import tqdm
import networkx as nx

from scipy.special import comb
import time
def get_topk_ap(idx,vecs,id2class):
    """
    计算ap
    Args:
        idx:向量索引
        vecs(NxD):向量矩阵维度 NxD
        id2class:
    """
    query = vecs[idx]
    all_sim = []
    for vec in vecs:
        sim = cos_sim(query,vec)
        all_sim.append(sim)
    all_sim = np.array(all_sim)
    sortidx=np.argsort(all_sim)[::-1]
    
    sim_id = np.array(id2class)[sortidx][1:]
    target_id = id2class[idx]
    result = sim_id==target_id
    ap = cal_apK(result)
    return ap

def cos_sim(v1,v2):
    # 余弦相似度
    return np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))

def cal_apK(result,k=2):
    ap = 0
    count = 0
    for i in range(k):
        count +=result[i]
        ap += count/(i+1)
    return ap/k

def get_map(word_vecs,id2class):
    ap_list = []
    for i in tqdm(range(len(id2class))):
        ap = get_topk_ap(i,word_vecs,id2class)
        ap_list.append(ap)
    return sum(ap_list)/len(ap_list)

def evaluate_clustering(cls_pred, cls_true):
    """ Evaluate clustering results

    :param cls_pred: a list of element lists representing model predicted clustering
    :type cls_pred: list
    :param cls_true: a list of element lists representing the ground truth clustering
    :type cls_true: list
    :return: a dictionary of clustering evaluation metrics
    :rtype: dict
    """
    # import pickle
    # pickle.dump({'pred':cls_pred,'true':cls_true},open('pred_true.pkl','wb'))

    vocab_pred = set(itertools.chain(*cls_pred))
    vocab_true = set(itertools.chain(*cls_true))
    assert (vocab_pred ==
            vocab_true), "Unmatched vocabulary during clustering evaluation"

    # Cluster number
    num_of_predict_clusters = len(cls_pred)

    # Cluster size histogram
    cluster_size2num_of_predicted_clusters = Counter(
        [len(cluster) for cluster in cls_pred])

    # Exact cluster prediction
    pred_cluster_set = set([frozenset(cluster) for cluster in cls_pred])
    gt_cluster_set = set([frozenset(cluster) for cluster in cls_true])
    num_of_exact_set_prediction = len(
        pred_cluster_set.intersection(gt_cluster_set))  # 计算cluster的重合

    # Clustering metrics
    word2rank = {}  # 单词index对应的 rank
    wordrank2gt_cluster = {}
    rank = 0
    for cid, cluster in enumerate(cls_true):
        for word in cluster:
            if word not in word2rank:
                word2rank[word] = rank
                rank += 1
            # 对词对应的rank给予聚类类别 cid是聚类类别
            wordrank2gt_cluster[word2rank[word]] = cid
    gt_cluster_vector = [ele[1] for ele in sorted(wordrank2gt_cluster.items())]

    wordrank2pred_cluster = {}
    for cid, cluster in enumerate(cls_pred):
        for word in cluster:
            wordrank2pred_cluster[word2rank[word]] = cid
    pred_cluster_vector = [ele[1]
                           for ele in sorted(wordrank2pred_cluster.items())]

    # print("pred_cluster_vector:", pred_cluster_vector)
    # print("gt_cluster_vector:", gt_cluster_vector)
    ARI = adjusted_rand_score(gt_cluster_vector, pred_cluster_vector)
    FMI = fowlkes_mallows_score(gt_cluster_vector, pred_cluster_vector)
    NMI = normalized_mutual_info_score(gt_cluster_vector, pred_cluster_vector)

    # Pair-based clustering metrics
    '''
    p: 0 0 1 1 1
    t: 0 0 0 1 1
    '''
    def pair_set(labels):
        S = set()
        cluster_ids = np.unique(labels)  # 聚类种类
        for cluster_id in cluster_ids:
            cluster = np.where(labels == cluster_id)[0]
            n = len(cluster)  # number of elements in this cluster
            if n >= 2:
                # 找出所有的cluster instance的排列组合
                for i in range(n):
                    for j in range(i + 1, n):
                        S.add((cluster[i], cluster[j]))
            # elif n > 0:
            #     S.add(cluster[0])

        return S

    F_S = pair_set(gt_cluster_vector)  # tp +tn
    F_K = pair_set(pred_cluster_vector)  # tp +fp
    # compute p r f1
    if len(F_K) == 0:
        pair_recall = 0
        pair_precision = 0
        pair_f1 = 0
    else:
        common_pairs = len(F_K & F_S)  # tp
        pair_recall = common_pairs / len(F_S)
        pair_precision = common_pairs / len(F_K)
        eps = 1e-6
        pair_f1 = 2 * pair_precision * pair_recall / \
            (pair_precision + pair_recall + eps)

    # KM matching
    mwm_jaccard = end2end_evaluation_matching(cls_true, cls_pred)
    eval = Evaluator(gt_cluster_vector)
    p, r, f = eval.calc_paired_f_measure(pred_cluster_vector)
    h, c, v = eval.calc_v_measure(pred_cluster_vector)

    metrics = {"ARI": ARI, "FMI": FMI, "NMI": NMI, "pair_recall": pair_recall, "pair_precision": pair_precision,
               "pair_f1": pair_f1, "predicted_clusters": cls_pred, "num_of_predicted_clusters": num_of_predict_clusters,
               "cluster_size2num_of_predicted_clusters": cluster_size2num_of_predicted_clusters,
               "num_of_exact_set_prediction": num_of_exact_set_prediction,
               "maximum_weighted_match_jaccard": mwm_jaccard,
               'feng_p': p,
               'feng_r': r,
               'feng_f': f,
               'homogeneity': h,
               'completeness': c,
               'v_measure_score': v, }

    return metrics


def end2end_evaluation_matching(groundtruth, result):
    """ Evaluate the maximum weighted jaccard matching of groundtruth clustering and predicted clustering

    :param groundtruth: a list of element lists representing the ground truth clustering
    :type groundtruth: list
    :param result: a list of element lists representing the model predicted clustering
    :type result: list
    :return: best matching score
    :rtype: float
    """
    n = len(groundtruth)
    m = len(result)
    G = nx.DiGraph()
    S = n + m
    T = n + m + 1
    C = 1e8
    for i in range(n):
        for j in range(m):
            s1 = groundtruth[i]
            s2 = result[j]
            s12 = set(s1) & set(s2)
            weight = len(s12) / (len(s1) + len(s2) - len(s12))
            weight = int(weight * C)
            if weight > 0:
                G.add_edge(i, n + j, capacity=1, weight=-weight)
    for i in range(n):
        G.add_edge(S, i, capacity=1, weight=0)
    for i in range(m):
        G.add_edge(i + n, T, capacity=1, weight=0)
    mincostFlow = nx.algorithms.max_flow_min_cost(G, S, T)
    mincost = nx.cost_of_flow(G, mincostFlow) / C
    return -mincost / m

class Evaluator(object):
    def __init__(self, gt):
        self.gt = gt

    def calc_v_measure(self, preds):
        return homogeneity_score(self.gt, preds), completeness_score(self.gt, preds), v_measure_score(self.gt, preds)

    def calc_paired_f_measure(self, preds):
        pred_n_clusters = len(set(preds))
        gt_cluster_num = len(set(self.gt))

        gt_n = 0.0
        gt = np.array(self.gt)
        for cluster in range(gt_cluster_num):
            gt_n += comb(len(np.where(gt == cluster)
                             [0]), 2, exact=True)  # 计算排列数量

        pred_n = 0.0
        pred_right_n = 0.0
        preds = np.array(preds)
        max_cluster = -1
        for cluster in range(pred_n_clusters):
            indices = np.where(preds == cluster)[0]
            if len(indices) > max_cluster:
                max_cluster = len(indices)
            pred_n += comb(len(indices), 2, exact=True)  # 计算排序组合

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if self.gt[indices[i]] == self.gt[indices[j]]:
                        pred_right_n += 1

        if pred_n == 0:
            p = 0
            r = 0
            f1 = 0
        else:
            p = pred_right_n / pred_n
            r = pred_right_n / gt_n
            if p == 0 and r == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
        # return pred_right_n, pred_n, gt_n, p, r, 2 * p * r / (p + r), max_cluster
        return p, r, f1