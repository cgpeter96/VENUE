import torch
import math
import numpy as np
import random
import sys
import os
import pickle
import shutil
sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
from misc.utils import (save_model, load_model, my_logger,
                        load_embedding, load_raw_data, Results, Metrics, read_synset, save_pkl)
from tqdm import tqdm
MAX_LENGTH = 25
random.seed(5417)
np.random.seed(5417)


def load_visual_synset(word2index, synset_path, use_npy=False):
    """读取视觉特征

    Args:
        word2index:
        synset_path:
        use_npy:
    """
    visual_dict = dict()
    visual_dict_path = dict()
    is_save = False
    fp = 'visual_dict.pkl'
    fp_p = 'visual_dict_path'
    if use_npy and os.path.exists(fp):
        print('load ', fp)
        with open(fp, 'rb') as f:
            visual_dict = pickle.load(f)
        if len(visual_dict.keys()) < 1:
            raise Exception("visual dict is empty")
        with open(fp_p, 'rb') as f:
            visual_dict_path = pickle.load(f)
        if len(visual_dict_path.keys()) < 1:
            raise Exception("visual_dict_path is empty")
        return {'visual_dict': visual_dict, 'visual_dict_path': visual_dict_path, 'use_npy': use_npy}

    elif os.path.exists(fp_p):
        print('load', fp_p)
        with open(fp_p, 'rb') as f:
            visual_dict_path = pickle.load(f)
        if len(visual_dict_path.keys()) < 1:
            raise Exception("visual_dict_path is empty")
        return {'visual_dict': visual_dict, 'visual_dict_path': visual_dict_path, 'use_npy': use_npy}

    else:
        is_save = True

    for word_path in tqdm(os.listdir(synset_path), desc='load visual synset'):
        word = word_path.split('(')[0].replace(' ', '_').split('.npy')[0]
        visual_dict_path[word2index[word]
                         ] = os.path.join(synset_path, word_path)
        if use_npy and is_save:
            visual_dict[word2index[word]] = np.load(
                os.path.join(synset_path, word_path))

    if is_save and use_npy:
        print('saving ', fp)
        with open(fp, 'wb') as f:
            pickle.dump(visual_dict, f)
        with open(fp_p, 'wb') as f:
            pickle.dump(visual_dict_path, f)
    return {'visual_dict': visual_dict, 'visual_dict_path': visual_dict_path, 'use_npy': use_npy}


class ElementDataset(Dataset):
    def __init__(self,
                 data,
                 visual_info=None,):
        self.sets_data, self.insts_data, self.labels_data = self.split_data(
            data)
        self.use_visual = False
        # print(visual_info)
        if visual_info is not None:
            print('load visual info')
            self.use_npy = visual_info['use_npy']
            if self.use_npy:
                self.visual_dict = visual_info['visual_dict']
            self.visual_dict_path = visual_info['visual_dict_path']

            self.use_visual = True

    def split_data(self, data):
        sets_data = []
        insts_data = []
        labels_data = []
        if isinstance(data, list):
            for i in data:
                sets_data.append(i['set'])
                insts_data.append(i['inst'])
                labels_data.append(i['label'])
            sets_data = torch.cat(sets_data, dim=0)
            insts_data = torch.cat(insts_data, dim=0)
            labels_data = torch.cat(labels_data, dim=0)
        else:
            sets_data = data['set']
            insts_data = data['inst']
            labels_data = data['label']

        return sets_data, insts_data, labels_data

    def get_visual_word(self, word_index, rows=50):
        """
        读取预提取的特征
        """
        if isinstance(word_index, torch.Tensor):
            word_index = word_index.item()
        if not self.use_npy:
            return np.load(self.visual_dict_path[word_index])[:rows]
        else:
            return self.visual_dict[word_index][:rows]

    def get_data(self, index):
        word_sets = self.sets_data[index]
        word_inst = self.insts_data[index]
        label = self.labels_data[index]

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, index):
        """

        Returns:
            visual_sets:np.array shape:(1,25)
            visual_inst:np.array shape:(1,1)
            word_sets: torch.tensor Size([25])
            word_inst:torch.tensor:Size([1])
            label:torch.tensor  size:Size([1])
        """
        word_sets = self.sets_data[index]
        word_inst = self.insts_data[index]
        label = self.labels_data[index]
        if self.use_visual:
            visual_sets = [np.expand_dims(
                self.get_visual_word(s), axis=0) for s in word_sets if s != 0]
            visual_inst = np.expand_dims(
                self.get_visual_word(word_inst), axis=0)
            visual_sets = np.vstack(visual_sets)
            return visual_sets, visual_inst, word_sets, word_inst, label

        return word_sets, word_inst, label


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
    word_sets_batch = torch.stack(word_sets)
    word_insts_batch = torch.stack(word_insts)
    labels_batch = torch.stack(labels)

    visual_sets_batch = np.vstack(visual_sets_pad)
    visual_insts_batch = np.vstack(visual_insts)
    return torch.from_numpy(visual_sets_batch).float(), torch.from_numpy(visual_insts_batch).float(), word_sets_batch, word_insts_batch, labels_batch


def get_element_dataloader(data, visual_info=None, collect_type='multimodal', batch_size=32, shuffle=False, num_workers=2):
    clt_fns = {
        'word': word_collect_fn,
        'multimodal': multimodal_collect_fn,
    }
    dataset = ElementDataset(data, visual_info)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=clt_fns[collect_type], pin_memory=True, num_workers=num_workers)
    return loader
