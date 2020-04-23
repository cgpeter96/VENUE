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
from misc.utils import load_embedding, read_synset
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import permutations

MAX_LENGTH = 25
random.seed(5417)
np.random.seed(5417)


def sample_data(query_set, vocab, neg_size):
    triplet_data = []
    if len(query_set) == 1:
        query = query_set[0]
        neg_set = set(vocab) - set(query_set)
        for neg in np.random.choice(list(neg_set), neg_size, replace=False):
            triplet_data.append([query, query, neg])
    else:
        for query in query_set:
            pos_set = set(query_set) - {query}
            neg_set = set(vocab) - set(query_set)
            for pos in pos_set:
                for neg in np.random.choice(list(neg_set), neg_size, replace=False):
                    triplet_data.append([query, pos, neg])
    return triplet_data


class RankDataset(Dataset):
    def __init__(self,
                 options,
                 word_synset_path,
                 visual_synset_path,
                 neg_size=25,
                 output_mode="visual",
                 mode='train',
                 use_npy=False):
        self.index2word = options["index2word"]
        self.word2index = options["word2index"]
        self.options = options
        self.vocab = []  # 词表
        self.positive_sets = []
        self.word_synset_path = word_synset_path
        self.visual_synset_path = visual_synset_path
        self.train_data = []
        self.neg_size = neg_size
        self.output_mode = output_mode
        self.visual_dict = None  # 设置为cache 形式
        self.visual_dict_path = None
        self.load_word_synset(self.word_synset_path)
        print(len(self.vocab))
        if self.output_mode == 'visual' or self.output_mode == 'multimodal':
            self.load_visual_synset(self.visual_synset_path, use_npy)

        if mode == 'train':
            self.get_pos_neg_data(self.neg_size)
        print("data size:", len(self.train_data))

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

        self.vocab = sorted(self.vocab)  # index

    def load_visual_synset(self, synset_path, use_npy=False):
        # TODO 缓存结构
        # self.visual_dict = dict()
        if use_npy:
            self.visual_dict = dict()
            self.visual_dict_path = dict()
            # 直接读取npy文件到内存
            for word in tqdm(os.listdir(synset_path)):
                full_word_path = os.path.join(synset_path, word)
                feat = np.load(full_word_path)
                word = word.split('(')[0].replace(' ', '_').split('.npy')[0]
                if word in self.index2word:
                    self.visual_dict_path[word] = full_word_path
                    self.visual_dict[word] = feat
            print('use npy feat')
        else:
            self.visual_dict_path = dict()
            for word in tqdm(os.listdir(synset_path)):
                full_word_path = os.path.join(synset_path, word)
                word = word.split('(')[0].replace(' ', '_').split('.npy')[0]
                if word in self.index2word:
                    self.visual_dict_path[word] = full_word_path

    def get_visual_feat(self, word_idx):
        """返回图像特征
        """
        word = self.index2word[word_idx]
        if self.visual_dict is not None:
            return self.visual_dict[word]

        elif self.visual_dict_path is not None:

            return np.load(self.visual_dict_path[word])
        else:

            raise Exception(
                "visual_dict_path&visual_dict is None,please initite those")

    def get_pos_neg_data(self, neg_size=10):
        # 最简单的random sample 策略 pos:neg = 1:neg_size

        pool = Pool(processes=8)
        print("start sampling")
        all_task = []
        for query_set in self.positive_sets:
            task = pool.apply_async(
                sample_data, (query_set, self.vocab, neg_size))
            all_task.append(task)
        pool.close()
        pool.join()
        for future in all_task:
            triplet_data = future.get()
            self.train_data.extend(triplet_data)
        print("sample done!")

    def __len__(self):
        return len(self.train_data)

    def get_vocab_tensor(self,word):
        # word = self.vocab[index]
        word_name = self.index2word[word] 
        if self.output_mode == "visual":
            visual_feat = self.get_visual_feat(word)
            return word_name,torch.tensor(visual_feat).float()

        elif self.output_mode == 'multimodal':
            visual_feat = self.get_visual_feat(word)
            return word_name,torch.tensor(word).long(),torch.tensor(visual_feat).float()
        else:

            return word_name, torch.tensor(word).long()

    def __getitem__(self, index):
        query, pos, neg = self.train_data[index]
        if self.output_mode == 'visual':
            query_emb = self.get_visual_feat(query)
            pos_emb = self.get_visual_feat(pos)
            neg_emb = self.get_visual_feat(neg)
            return torch.tensor(query_emb), torch.tensor(pos_emb), torch.tensor(neg_emb)

        elif self.output_mode == 'multimodal':
            query_emb = self.get_visual_feat(query)
            pos_emb = self.get_visual_feat(pos)
            neg_emb = self.get_visual_feat(neg)
            return torch.tensor(query).long(), torch.tensor(pos).long(), torch.tensor(neg).long(), torch.tensor(query_emb), torch.tensor(pos_emb), torch.tensor(neg_emb)

        else:
            query, pos, neg = self.train_data[index]
            return torch.tensor(query).long(), torch.tensor(pos).long(), torch.tensor(neg).long()


class BatchRankDataset(Dataset):
    def __init__(self,
                 options,
                 word_synset_path,
                 visual_synset_path,
                 output_mode="visual",
                 mode='train',
                 use_npy=False):
        self.index2word = options["index2word"]
        self.word2index = options["word2index"]
        self.vocab = []  # 词表
        self.positive_sets = []
        self.word_synset_path = word_synset_path
        self.visual_synset_path = visual_synset_path
        self.output_mode = output_mode
        self.visual_dict = None  # 设置为cache 形式
        self.visual_dict_path = None
        self.index2label = {} # 映射标签用
        self.pair_data = [] # pair data
        self.mode = mode
        self.load_word_synset(self.word_synset_path)
        if self.output_mode == 'visual' or self.output_mode == 'multimodal':
            self.load_visual_synset(self.visual_synset_path, use_npy)

    def load_word_synset(self, synset_path):
        """
        读取文本同义词集合,c

        :synset_path: the path of synset
        :type synset_path: str
        :return :None
        :rtype: None
        """
        self.synset = read_synset(synset_path)
        if self.mode=="train":
            random.shuffle(self.synset)
        for line in tqdm(self.synset, desc='loading word synset...'):
            words = sorted([self.word2index[word]for word in line])
            pairs = list(permutations(words,2))
            self.pair_data.extend(pairs)
            self.positive_sets.append(words)
            self.vocab.extend(words)

        self.vocab = sorted(self.vocab)  # index

        for idx,sets in enumerate(self.positive_sets):
            for widx in sets:
                self.index2label[widx]=idx


    def load_visual_synset(self, synset_path, use_npy=False):
        # TODO 缓存结构
        # self.visual_dict = dict()
        if use_npy:
            self.visual_dict = dict()
            self.visual_dict_path = dict()
            # 直接读取npy文件到内存
            for word in tqdm(os.listdir(synset_path)):
                full_word_path = os.path.join(synset_path, word)
                feat = np.load(full_word_path)
                word = word.split('(')[0].replace(' ', '_').split('.npy')[0]
                if word in self.index2word:
                    self.visual_dict_path[word] = full_word_path
                    self.visual_dict[word] = feat
            print('use npy feat')
        else:
            self.visual_dict_path = dict()
            for word in tqdm(os.listdir(synset_path)):
                full_word_path = os.path.join(synset_path, word)
                word = word.split('(')[0].replace(' ', '_').split('.npy')[0]
                if word in self.index2word:
                    self.visual_dict_path[word] = full_word_path

    def get_visual_feat(self, word_idx):
        """返回图像特征
        """
        word = self.index2word[word_idx]
        if self.visual_dict is not None:
            return self.visual_dict[word]

        elif self.visual_dict_path is not None:

            return np.load(self.visual_dict_path[word])
        else:

            raise Exception(
                "visual_dict_path&visual_dict is None,please initite those")

    def __len__(self):
        return len(self.pair_data)

    def get_vocab_tensor(self,word):
        """ get one word feat 
        """
        # word = self.vocab[index]
        word_name = self.index2word[word] 
        if self.output_mode == "visual":
            visual_feat = self.get_visual_feat(word)
            return word_name,torch.tensor(visual_feat).float()

        elif self.output_mode == 'multimodal':
            visual_feat = self.get_visual_feat(word)
            return word_name,torch.tensor(word).long(),torch.tensor(visual_feat).float()
        else:

            return word_name, torch.tensor(word).long()


    def __getitem__bk(self, index):

        query = self.vocab[index]
        label = self.index2label[query]
        if self.output_mode == 'visual':
            query_emb = self.get_visual_feat(query)
            return torch.tensor(query_emb)

        elif self.output_mode == 'multimodal':
            query_emb = self.get_visual_feat(query)

            return torch.tensor(label).long(), torch.tensor(query_emb),
        else:
    
            return torch.tensor(label).long(),

    def __getitem__(self, index):
        pairs = self.pair_data[index]
        pairs_label = [self.index2label[p] for p in pairs]

        if self.output_mode == 'visual':
            pair_query_emb = [self.get_visual_feat(query)[np.newaxis,:] for  query in pairs ] 
            pair_query_emb = np.vstack(pair_query_emb)
            return torch.tensor(pairs_label).long(), torch.from_numpy(pair_query_emb)

        elif self.output_mode == 'multimodal':
            pair_query_emb = [self.get_visual_feat(query)[np.newaxis,:] for  query in pairs ] 
            pair_query_emb = np.vstack(pair_query_emb)
            return torch.tensor(pairs_label).long(), torch.tensor(pairs).long(),torch.from_numpy(pair_query_emb)

        else:
    
            return torch.tensor(pairs_label).long(), torch.tensor(pairs).long()


def text_collate_fn(data):
    labels,text = zip(*data)
    labels = torch.cat(labels,dim=0)
    text = torch.cat(text,dim=0)
    return labels,text

def visual_collate_fn(data):
    labels,visual = zip(*data)
    labels = torch.cat(labels,dim=0)
    visual = torch.cat(visual,dim=0)
    return labels,visual

def multimodal_collate_fn(data):
    labels,text,visual = zip(*data)
    labels = torch.cat(labels,dim=0)
    text = torch.cat(text,dim=0)
    visual = torch.cat(visual,dim=0)
    return labels,text,visual



if __name__ == '__main__':
    options = {}
    options["dataset"] = 'NICE_fast'
    options["data_format"] = 'set'
    fi = "/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/wiki_unique_std.txt.vec"
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(
        fi)
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size

    dataset = BatchRankDataset(options=options,
                             word_synset_path='/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/wiki_std_synset/train_label.txt',
                             # "/home/chenguang/workhome/nice_tag_data/synset_images_feat_pack_7",synset_images_np100_trans,
                             visual_synset_path='/home/chenguang/workhome/nice_tag_data/wiki_data_feat2048',
                             output_mode="multimodal",
                             mode='train',
                             use_npy=False
                             )

    # print(len(dataset))
    # exit()
    loader = dataloader.DataLoader(dataset, batch_size=32, num_workers=4,collate_fn=multimodal_collate_fn)
    l = len(loader)
    import time
    gap = time.time()
    for idx, batch in enumerate(loader):
        a = batch
        print(a[-1].shape)
        # print(a[0].shape)
        if idx % 10 == 0:
            print(time.time() - gap)
            gap = time.time()
            print(idx, '/', l, a[0].shape)

