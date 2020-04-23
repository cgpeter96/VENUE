import torch
import math
import numpy as np
import random
import sys
import os
import pickle
import shutil
sys.path.append('..')
from torch.utils.data import Dataset, dataloader
from misc.utils import (save_model, load_model, my_logger,
                        load_embedding, load_raw_data, Results, Metrics, read_synset, save_pkl)
from tqdm import tqdm

MAX_LENGTH = 25
random.seed(5417)
np.random.seed(5417)

def prepare_data(path, target_path):
    for word_path in tqdm(os.listdir(path)):
        full_word_path = os.path.join(path, word_path)
        img_list = []
        for file in os.listdir(full_word_path):
            img = np.expand_dims(
                np.load(os.path.join(full_word_path, file)), axis=0)
            img_list.append(img)
        imgs = np.vstack(img_list)
        print(imgs.shape)
        full_target_path = os.path.join(target_path, word_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        np.save(full_target_path, imgs)

# -----------train-test-------------


class NICESynsetDataset(Dataset):
    """
    基于Nice的多模态同义词数据集
    """

    def __init__(self,
                 options,
                 word_synset_path,
                 visual_synset_path,
                 max_set_lengths=25,  # set最大长度
                 neg_sample_size=5,  # 每个正例对应的负例个数
                 mode='train',
                 use_visual=True,  # 是否使用图像特征
                 use_npy=False,  # 选择使用npy还是full path
                 visual_info = None,# 
                 ):
        self.word_synset_path = word_synset_path
        self.visual_synset_path = visual_synset_path
        self.options = options
        self.index2word = options["index2word"]
        self.word2index = options["word2index"]

        self.max_set_lengths = max_set_lengths
        self.neg_sample_size = neg_sample_size  # 每个set所对应的抽样负例的数量
        self.vocab = []  # 词表
        self.positive_sets = []  # ground truth
        self.visual_dict = None
        self.visual_dict_path = None
        # 训练集数据
        self.train_sets = []
        self.train_insts = []
        self.train_labels = []
        self.mode = mode
        self.use_npy = use_npy
        self.use_visual = use_visual
        self.load_word_synset(self.word_synset_path)

        if visual_info:
            self.visual_dict = visual_info['visual_dict']
            self.visual_dict_path = visual_info['visual_dict_path']

        if visual_info is None and self.use_visual:
            self.load_visual_synset(
                self.visual_synset_path, self.use_npy)
            # print(self.visual_dict.keys())
        if self.mode == 'train':
            self.generate_data(1)
            pos = sum(self.train_labels)
            neg = len(self.train_labels) - pos
            print('train set pos:{},neg:{}'.format(pos, neg))
        elif self.mode == 'test':
            self.neg_sample_size = 10
            self.generate_data(1)
            pos = sum(self.train_labels)
            neg = len(self.train_labels) - pos
            print('test set pos:{},neg:{}'.format(pos, neg))

    def __len__(self):
        return len(self.train_labels)

    def get_test_by_word(self, word):
        word_inst = np.array(word)
        if self.use_visual:
            visual_inst = self.load_visual_word(word)  #
            # 1,1
            return torch.from_numpy(visual_inst), torch.from_numpy(word_inst).long()
        else:
            return torch.from_numpy(word_inst).long()

    def __getitem__(self, index):
        sets = self.train_sets[index]
        # word_sets = np.zeros(self.max_set_lengths)
        # word_sets[:len(sets)] = sets
        word_sets = sets
        word_inst = self.train_insts[index]
        label = self.train_labels[index]
        if self.visual_dict_path is not None:
            # TODO 可以控制使用多少张图片 现在是50
            visual_sets = [np.expand_dims(
                self.load_visual_word(s), axis=0) for s in sets]
            visual_inst = np.expand_dims(
                self.load_visual_word(word_inst), axis=0)
            visual_sets = np.vstack(visual_sets)
            return visual_sets, np.array(visual_inst), np.array(word_sets), np.array(word_inst), np.array(label)
        return torch.tensor(word_sets).long(), torch.tensor(word_inst).long(), torch.tensor(label).long()

    def load_word_synset(self, synset_path):
        """
        读取文本同义词集合,c

        :synset_path: the path of synset
        :type synset_path: str
        :return :None
        :rtype: None
        """
        text = read_synset(synset_path)
        random.shuffle(text)
        for line in tqdm(text, desc='loading word synset...'):
            words = sorted([self.word2index[word]for word in line])
            self.positive_sets.append(words)
            self.vocab.extend(words)

        self.vocab = sorted(self.vocab)
        # for sets in self.positive_sets:
        #     self.vocab.extend(sets)

    def load_visual_word(self, word_path, rows=50):
        """
        读取预提取的特征
        """
        if not self.use_npy:
            return np.load(self.visual_dict_path[word_path])[:rows]
        else:
            return self.visual_dict[word_path][:rows]

    def load_visual_synset(self, synset_path, use_npy=False):
        """
        构建视觉同义词路径词典
        """
        self.visual_dict = dict()
        self.visual_dict_path = dict()
        is_save = False
        fp = 'visual_dict.pkl'
        if use_npy and os.path.exists(fp):
            print('load ', fp)
            with open(fp, 'rb') as f:
                self.visual_dict = pickle.load(f)
            if len(self.visual_dict.keys()) < 1:
                raise Exception("visual dict is empty")
            return None
            # if len(self.visual_dict.keys()) < 1:
            #     is_save = True
            # else:
            #     return None
        else:
            is_save = True

        for word_path in tqdm(os.listdir(synset_path), desc='load visual synset'):
            word = word_path.split('(')[0].replace(' ', '_').split('.npy')[0]
            self.visual_dict_path[self.word2index[word]
                                  ] = os.path.join(synset_path, word_path)
            if use_npy and is_save:
                self.visual_dict[self.word2index[word]] = np.load(
                    os.path.join(synset_path, word_path))

        if is_save and use_npy:
            print('saving ', fp)
            with open(fp, 'wb') as f:
                pickle.dump(self.visual_dict, f)

    def get_negative_samples_pool(self, synset_words):
        """
        返回抽取的负例词pool
        """
        pos_set = set(synset_words)
        all_set = set(self.vocab)
        return list(all_set - pos_set)

    def sample_negative_data(self, raw_set, neg_sample_size):
        """
        根据给定的同义词集合生成一定数量的正负例样本
        """
        neg_sample_pool = self.get_negative_samples_pool(raw_set)
        sub_size = len(raw_set)

        sample_sets = []
        sample_insts = []
        sample_labels = []

        def get_pos_candidates(raw_set, subset_size):
            """从同义词中获取正例集合,

            :return candidates: 正例集合 [ set集合:list，pos_inst集合:list]
            ：rtype candidates:list
            """
            candidates = []
            random.shuffle(raw_set)
            for size in range(1, subset_size, 2):
                if size > 10:
                    break
                for i in range(len(raw_set)):
                    if size >= len(raw_set):
                        candidates.append([raw_set, []])
                        break
                    if i + size <= len(raw_set):
                        subset = raw_set[i:i + size]
                        random.shuffle(subset)  # 对subset进行shuffle 更好学习 置换不变性
                        pos_insts = list(set(raw_set) - set(subset))
                        candidates.append([subset, pos_insts])
            return candidates

        candidates = get_pos_candidates(raw_set, len(raw_set))

        if True:
            """
            fix pair negative sample size
            """
            for candidate in candidates:
                sets, pos_insts = candidate

                if len(pos_insts) > 0:
                    for pos_inst in pos_insts:
                        # data.append([sets, pos_inst])
                        sample_sets.append(sets)
                        sample_insts.append(pos_inst)
                        sample_labels.append(1)

                    neg_insts = np.random.choice(
                        neg_sample_pool, size=neg_sample_size)
                    for neg_inst in neg_insts:
                        sample_sets.append(sets)
                        sample_insts.append(neg_inst)
                        sample_labels.append(0)
                else:
                    neg_insts = np.random.choice(
                        neg_sample_pool, size=neg_sample_size)
                    for neg_inst in neg_insts:
                        sample_sets.append(sets)
                        sample_insts.append(neg_inst)
                        sample_labels.append(0)

        else:
            pass

        return sample_sets, sample_insts, sample_labels

    def sample_data(self):
        sample_sets = []
        sample_insts = []
        sample_labels = []

        for raw_set in self.positive_sets:
            if len(raw_set) < 2:
                continue

            sets, insts, labels = self.sample_negative_data(
                raw_set, self.neg_sample_size)

            sample_sets.extend(sets)
            sample_insts.extend(insts)
            sample_labels.extend(labels)
        return sample_sets, sample_insts, sample_labels

    def sample_pos_neg_data(self, raw_sets, max_length, neg_sample_size):
        """
        same with synsetmine
        """
        sip_triplets = []
        pos_sip_cnt_sum = 0
        neg_sip_cnt_sum = 0

        sample_sets = []
        sample_insts = []
        sample_labels = []

        for subset_size in range(1, max_length + 1):
            for raw_set in raw_sets:  # 每一个都是同义词集
                if len(raw_set) < subset_size:
                    continue
                raw_set_new = raw_set.copy()
                random.shuffle(raw_set_new)
                batch_set = []
                batch_pos = []
                if len(raw_set) == subset_size:  # put the entire full set
                    for _ in range(neg_sample_size + 1):
                        batch_set.append(raw_set)
                        batch_pos.append(random.choice(raw_set))
                else:
                    for _ in range(neg_sample_size + 1):
                        '''
                        subset_size = 1
                        raw_set_new = [a,b,c,d]

                        '''
                        for start_idx in range(0, len(raw_set_new) - subset_size, subset_size + 1):
                            # slide window= subset_size
                            subset = raw_set_new[start_idx:start_idx + subset_size]
                            # 取subset后一个词
                            pos_inst = raw_set_new[start_idx + subset_size]
                            batch_set.append(subset)
                            batch_pos.append(pos_inst)
                        random.shuffle(raw_set_new)

                pos_sip_cnt = int(len(batch_set) / (neg_sample_size + 1))
                pos_sip_cnt_sum += pos_sip_cnt
                neg_sip_cnt = int(pos_sip_cnt * neg_sample_size)
                neg_sip_cnt_sum += neg_sip_cnt

                negative_pool = [
                    ele for ele in self.vocab if ele not in raw_set]
                sample_size = math.gcd(
                    neg_sip_cnt, len(negative_pool))  # 最大公约数
                sample_times = int(neg_sip_cnt / sample_size)

                batch_neg = []
                for _ in range(sample_times):
                    batch_neg.extend(random.sample(
                        negative_pool, sample_size))
                    # 每次抽出相同大小的neg
                for idx, subset in enumerate(batch_set):

                    # 设置随机
                    # random.shuffle(subset)
                    '''
                    pos=np.random.choice(batch_pos)

                    sample_sets.append(subset)
                    sample_insts.append(pos)
                    sample_labels.append(1)
                    sip_triplets.append((subset, pos, 1))

                    neg = np.random.choice(batch_neg)
                    # random.shuffle(subset)
                    sip_triplets.append((subset, neg, 0))
                    sample_sets.append(subset)
                    sample_insts.append(neg)
                    sample_labels.append(0)

                    '''
                    if idx < pos_sip_cnt:
                        pos = batch_pos[idx]

                        # 设置随机
                        # random.shuffle(subset)
                        sample_sets.append(subset)
                        sample_insts.append(pos)
                        sample_labels.append(1)
                        sip_triplets.append((subset, pos, 1))
                    else:
                        neg = batch_neg[idx - pos_sip_cnt]
                        # random.shuffle(subset)
                        sip_triplets.append((subset, neg, 0))
                        sample_sets.append(subset)
                        sample_insts.append(neg)
                        sample_labels.append(0)
                    

        print(len(sip_triplets))
        return sample_sets, sample_insts, sample_labels

    def sample_random_walk(self, raw_sets, pos_per_sets,neg_per_sets):
        """
        使用随机游走生成数据
        """
        sample_sets = []
        sample_insts = []
        sample_labels = []
        
        def choise_by_step(start,ary,step):
            """
            形成随机游走结果
            """
            cur_idx=start
            
            seq = [ary[start]]
            cur_idx +=step.cumsum()
            for idx in cur_idx:
                
                if idx>=0 and idx<len(ary):
                    if ary[idx] not in seq:
                        seq.append(ary[idx])
                else:
                   continue
            return seq


        for idx,raw_set in enumerate(raw_sets):
            set_len = len(raw_set) 
            sets=[]
            for start in range(len(raw_set)):
                sets.append(choise_by_step(start,raw_set,np.random.choice([-1,1],set_len)))
            for s in sets:
                # 构造正例
                pos_insts = np.random.choice(raw_set,pos_per_sets) #构成正例 使用词长一半
                for pos in pos_insts:
                              
                    sample_sets.append(random.shuffle(s))
                    sample_insts.append(pos)
                    sample_labels.append(1)
                neg_set = raw_sets[:idx-1]+raw_sets[idx+1:idx+5]
                neg_pool = []
                for i in neg_set:
                    neg_pool.extend(i)
                neg_insts = np.random.choice(neg_pool,neg_per_sets)
                for neg in neg_insts:
                    sample_sets.append(s)
                    sample_insts.append(neg)
                    sample_labels.append(0)
        return sample_sets,sample_insts,sample_labels

    def generate_data(self,sample_iter=15,strategy='paper'):
        """
        生成数据
        """
        '''
        for raw_set in self.positive_sets:
            sample_sets, sample_insts, sample_labels = self.sample_negative_data(
                raw_set, self.neg_sample_size)
            
            self.train_sets.extend(sample_sets)
            self.train_insts.extend(sample_insts)
            self.train_labels.extend(sample_labels)
        '''
        # random.shuffle(self.positive_sets)
        print(len(self.positive_sets))
        for i in range(sample_iter):
            if strategy=='paper':
                # paper strategy
                # best performance 0.27
                sample_sets, sample_insts, sample_labels = self.sample_pos_neg_data(
                    self.positive_sets, self.max_set_lengths, self.neg_sample_size)
            elif strategy=='walk':
                # random walk strategy
                # best performance 0.24
                sample_sets, sample_insts, sample_labels = self.sample_random_walk(
                    self.positive_sets,5,10)
            else:
                sample_sets,sample_insts,sample_labels = self.sample_data()
            self.train_sets.extend(sample_sets)
            self.train_insts.extend(sample_insts)
            self.train_labels.extend(sample_labels)
        return self.train_sets, self.train_insts, self.train_labels


def word_collect_fn(data):
    word_sets, word_insts, labels = zip(*data)
    lens = len(labels)
    max_length = MAX_LENGTH  # max([len(i) for i in word_sets])  # MAX_LENGTH
    word_sets_pad = torch.zeros(lens, max_length)

    for i, sets in enumerate(word_sets):
        end = len(sets)
        word_sets_pad[i, :end] = sets
    word_insts_batch = torch.stack(word_insts).view(-1, 1)
    labels_batch = torch.stack(labels).view(-1, 1)
    return word_sets_pad.long(), word_insts_batch.long(), labels_batch.float()


def multimodal_collect_fn(data):
    """
    拼接dataset的输出结果
    """
   # zero_img = np.zeros([1, 25, 128, 7, 7])
    zero_word = np.zeros(1)
    visual_sets, visual_insts, word_sets, word_insts, labels = zip(*data)
    zero_img = np.zeros_like(visual_insts[0])
    lens = len(labels)
    max_length = MAX_LENGTH  # max([len(i) for i in word_sets])
    visual_sets_pad = []
    word_sets_pad = []
    for sets in visual_sets:
        if sets.shape[0] < max_length:
            count = max_length - sets.shape[0]
            pad = np.repeat(zero_img, repeats=count, axis=0)
            sets_pad = np.expand_dims(np.vstack([sets, pad]), axis=0)
        else:
            sets_pad = np.expand_dims(sets, axis=0)
        visual_sets_pad.append(sets_pad)
    for sets in word_sets:
        if sets.shape[0] < max_length:
            count = max_length - sets.shape[0]
            pad = np.repeat(zero_word, repeats=count, axis=0)
            sets_pad = np.expand_dims(np.hstack([sets, pad]), axis=0)
        else:
            sets_pad = np.expand_dims(sets, axis=0)
        word_sets_pad.append(sets_pad)

    visual_sets_batch = np.vstack(visual_sets_pad)
    word_sets_batch = np.vstack(word_sets_pad)
    visual_insts_batch = np.vstack(visual_insts)
    word_insts_batch = np.array(word_insts).reshape(lens, -1)
    labels_batch = np.array(labels).reshape(lens, -1)
    return torch.from_numpy(visual_sets_batch).float(), torch.from_numpy(visual_insts_batch).float(), torch.from_numpy(word_sets_batch).long(), torch.from_numpy(word_insts_batch).long(), torch.from_numpy(labels_batch).long()


#------------feature-----------------


class FeatureDataset(Dataset):
    def __init__(self, vocab_dataset):
        self.vocab_dataset = vocab_dataset
        self.use_visual = self.vocab_dataset.use_visual
        self.vocab = vocab_dataset.vocab

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, index):
        """
        output:
            visual 50,2048,7,7
            word  1

        """

        data = self.vocab_dataset.get_test_by_word(self.vocab[index])
        if self.use_visual:
            v, w = data
            w = w.view(1)
            return v, w
        else:
            w = data
            w = w.view(1)
            return w


def get_feature_loader(vocab_dataset, batch_size):
    """
    output :
        visual: batch,50,2048,7,7
        word batch
    """
    dataset = FeatureDataset(vocab_dataset)
    return dataloader.DataLoader(dataset, batch_size=batch_size)

# -----------cluster-----------------


class ClusterInstDataset(Dataset):
    def __init__(self, vocab_list, feat_list, clusters, inst):
        self.vocab_list = vocab_list
        self.feat_list = feat_list
        self.clusters = clusters  # list :[[1,2,3],...]
        self.inst = inst
        self.set_data = []
        self.inst_data = []
        self.vocab2feat = dict()

        self.initialize()

    def __len__(self):
        return len(self.inst_data)

    def __getitem__(self, index):
        word_sets = self.set_data[index]
        word_inst = self.inst_data[index]
        # print('->', word_sets)
        set_feat = [self.vocab2feat[s] for s in word_sets]

        inst_feat = self.vocab2feat[word_inst]
        set_feat = np.vstack(set_feat)
        return set_feat, inst_feat

    def initialize(self):
        for cluster in self.clusters:
            self.set_data.append(cluster)
            self.inst_data.append(self.inst)
        for idx, feat in zip(self.vocab_list, self.feat_list):
            self.vocab2feat[idx] = feat.reshape(1, -1)  # 1,228


def cluster_collect_fn(data):
    """
    将set feat和inst_feat打包成batch形式
    """
    sets_feats, insts_feats = zip(*data)
    zero_feat = np.zeros_like(insts_feats[0])
    lens = len(insts_feats)
    max_length = max([len(i) for i in sets_feats])
    mask = torch.zeros(len(insts_feats), max_length)
    for idx, sets in enumerate(sets_feats):
        end = len(sets)
        mask[idx][:end] = 1
    sets_pad_batch = []
    # print(len(sets_feats))
    for sets in sets_feats:
        # sets n,228 n<25
        if sets.shape[0] < max_length:
            count = max_length - sets.shape[0]
            pad = np.repeat(zero_feat, repeats=count, axis=0)
            sets_pad = np.expand_dims(np.vstack([sets, pad]), axis=0)
        else:

            sets_pad = np.expand_dims(sets, axis=0)
        # print(sets_pad.shapeq)
        sets_pad_batch.append(sets_pad)
    # save_pkl(sets_pad_batch,'sets_pad_batch.pkl')
    sets_pad_batch = np.vstack(sets_pad_batch)
    # print(sets_pad_batch.shape)
    insts_feats_batch = np.vstack(insts_feats)
    return torch.from_numpy(sets_pad_batch).float(), torch.from_numpy(insts_feats_batch).float(), mask.float()


def get_cluster_inst_loader(vocab_list, feat_list, clusters, inst, batch_size):
    cluster_dataset = ClusterInstDataset(
        vocab_list, feat_list, clusters, inst)
    return dataloader.DataLoader(cluster_dataset, batch_size=batch_size, collate_fn=cluster_collect_fn)


if __name__ == '__main__':

    # prepare_data('/home/cheng/data/nice_tag_data/synset_images_feat_7',
    #              '/home/cheng/data/nice_tag_data/synset_images_np')
    # exit()
    import random
    options = {}
    options["dataset"] = 'NICE_fast'
    options["data_format"] = 'set'
    fi = "/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/fast_vector.txt.vec"
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(
        fi)
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size

    dataset = NICESynsetDataset(options,
                                '/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/split_0/train_label.txt',
                                '/home/chenguang/workhome/nice_tag_data/synset_images_np128',
                                use_visual=False,
                                use_npy=True,
                                neg_sample_size=40, max_set_lengths=MAX_LENGTH)
    s,i,l=dataset.sample_random_walk(dataset.positive_sets,5,10)
    for idx in range(100,300):
        print('sets-->',[index2word[x]for x in s[idx]],',inst-->',index2word[i[idx]],',label-->',l[idx])


    '''
    print(len(dataset.train_labels))
    print(len([i for i in dataset.train_labels if i == 0]))
    print(len([i for i in dataset.train_labels if i == 1]))
    test_dataset = NICESynsetDataset(options,
                                     '/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/split_0/test_label.txt',
                                     '/home/chenguang/workhome/nice_tag_data/synset_images_np128',
                                     mode='test', use_visual=False, use_npy=True,
                                     max_set_lengths=MAX_LENGTH)
    '''
    # get_train_loader(test_dataset)
    # print("all:", len(test_dataset.train_labels))
    '''
    print(dataset.index2word[4084])
    print(len(dataset.train_labels))
    print(sum(dataset.train_labels))

    print(len(test_dataset.train_labels))
    print(sum(test_dataset.train_labels))
    loader = get_cluster_inst_loader(list(range(100)),np.random.rand(100,228),[[1,2,3],[4,5,6]],3,2)
    # print(len(test_dataset.vocab))

    # print(test_dataset.vocab)
    
    # loader = get_feature_loader(test_dataset, 10)
    '''
    
    '''
    def get_train_loader(dataset):
        loader = dataloader.DataLoader(
            dataset, shuffle=True, batch_size=1000, collate_fn=word_collect_fn)

        for i in range(50):
            s = 0  # 样本综述
            l = 0  # check set大于0的个数
            la = 0  # label >0的
            for idx, val in enumerate(loader):
                # a, b, c, d, e = val
                sets, insts, label = val
                if idx == 0:
                    print(sets[0])
                l0 = (sets > 0).sum()
                s += sets.size(0)
                l += l0
                la += (label > 0).sum()
            print("样本总数：", s)
            print('set 大于0的个数：', l)
            print('label等于1的个数：', la)

            # s, i, l =

        # mask = (c != 0).float().unsqueeze(-1)
        # print(mask.shape)
        # print(d.shape)
        # print(e.shape)

    get_train_loader(dataset)
    '''
