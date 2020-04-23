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
from utils import (save_model, load_model, my_logger,
                   load_embedding, load_raw_data, Results, Metrics, read_synset)
from tqdm import tqdm

np.random.seed(2019)
random.seed(2019)

MAX_LENGTH = 23


class ClusterInstDataset(Dataset):
    def __init__(self, vocab_dataset, clusters, inst, batch_size):
        self.clusters = clusters  # list :[[1,2,3],...]
        self.inst = inst
        self.batch_size = batch_size
        self.vocab_dataset = vocab_dataset
        self.set_data = []
        self.inst_data = []
        self.get_set_inst_pair()

    def __len__(self):
        return len(self.inst_data)

    def __getitem__(self, index):
        word_sets = self.set_data[index]
        word_inst = self.inst_data[index]
        if self.vocab_dataset.visual_dict_path is not None:
            visual_sets = [np.expand_dims(
                self.vocab_dataset.load_visual_word(s), axis=0) for s in word_sets]
            visual_inst = np.expand_dims(
                self.vocab_dataset.load_visual_word(word_inst), axis=0)
            visual_sets = np.vstack(visual_sets)
            return visual_sets, np.array(visual_inst), np.array(word_sets), np.array(word_inst)
        return torch.tensor(word_sets).long(), torch.tensor(word_inst).long()

    def get_set_inst_pair(self):
        for cluster in self.clusters:
            self.set_data.append(cluster)
            self.inst_data.append(self.inst)


def cluster_collect_fn(data):
    zero_word = np.zeros(1)
    visual_sets, visual_insts, word_sets, word_insts = zip(*data)
    zero_img = np.zeros_like(visual_insts[0])
    lens = len(word_insts)
    max_length = 25  # max([len(i) for i in word_sets])
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
    return (torch.from_numpy(visual_sets_batch).float(), torch.from_numpy(word_sets_batch).long(),
            torch.from_numpy(visual_insts_batch).float(), torch.from_numpy(word_insts_batch).long())
