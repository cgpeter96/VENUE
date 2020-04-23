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
import h5py


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()  # 构建cuda stream

        self.preload()

    def preload(self):
        try:
            self.fetch_data = next(self.loader)
        except StopIteration:
            self.fetch_data = None
            return
        with torch.cuda.stream(self.stream):
            self.fetch_data = [data.cuda(non_blocking=True)
                               for data in self.fetch_data]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        fetch_data = self.fetch_data
        self.preload()
        return fetch_data


class Hdf5Saver(object):
    """
    形成hdf5文件

    note:形成文件过于庞大可能无法运行

    """

    def __init__(self, raw_dataset, batch_size=32, output_size=5):
        self.raw_dataset = raw_dataset

        self.loader = dataloader.DataLoader(
            dataset, shuffle=False, batch_size=batch_size, collate_fn=multimodal_collect_fn, num_workers=2)
        self.data_lengths = len(self.loader)
        self.fp = h5py.File("multimodal_data.h5", "w")
        self.batch_size = batch_size
        self.output_size = output_size
        self.create_data()

    def create_data(self):

        for idx, data in enumerate(tqdm(self.loader)):
            vs, vi, ws, wi, lb = data
            break

        self.fp.create_dataset(
            "ws", [self.data_lengths] + list(ws.shape), np.float32)
        self.fp.create_dataset(
            "wi", [self.data_lengths] + list(wi.shape), np.float32)
        self.fp.create_dataset(
            "lb", [self.data_lengths] + list(lb.shape), np.float32)

        if self.output_size == 5:
            self.fp.create_dataset(
                "vs", [self.data_lengths] + list(vs.shape), np.float32)
            self.fp.create_dataset(
                "vi", [self.data_lengths] + list(vi.shape), np.float32)

    def write_hdf5(self):
        print(len(self.loader))
        count = 0
        for idx, data in enumerate(tqdm(self.loader)):
            data = [i.cpu().numpy() for i in data]
            if len(data[-1]) != self.batch_size:
                print(data[-1])
                continue
            # print(idx, count)
            if self.output_size == 5:
                vs, vi, ws, wi, lb = data

                self.fp['vs'][idx] = vs
                self.fp['vi'][idx] = vi
                self.fp['ws'][idx] = ws
                self.fp['wi'][idx] = wi
                self.fp['lb'][idx] = lb
                count += 1
            else:
                ws, wi, lb = data

                # print(wi.shape)
                self.fp['ws'][idx] = ws
                self.fp['wi'][idx] = wi
                self.fp['lb'][idx] = lb
                count += 1

        self.fp.create_dataset("index", (count,),
                               np.int32, np.arange(count))

        self.fp.close()


class Hdf5Dataset(Dataset):
    def __init__(self, filename):
        self.hdf5_file = h5py.File(filename, 'r')
        self.idxs = self.hdf5_file['index'].value

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        vs = self.hdf5_file['vs'][self.idxs[index]]
        vi = self.hdf5_file['vs'][self.idxs[index]]
        ws = self.hdf5_file['vs'][self.idxs[index]]
        wi = self.hdf5_file['vs'][self.idxs[index]]
        lb = self.hdf5_file['vs'][self.idxs[index]]
        vs = Hdf5Dataset.covert2tensor(vs)
        vi = Hdf5Dataset.covert2tensor(vi)
        ws = Hdf5Dataset.covert2tensor(ws)
        wi = Hdf5Dataset.covert2tensor(wi)
        lb = Hdf5Dataset.covert2tensor(lb)
        return vs.float(), vi.float(), ws.float(), wi.float(), lb.float()

    @staticmethod
    def covert2tensor(np_data):
        return torch.from_numpy(np_data)


if __name__ == '__main__':
    from dataset import *
    # prepare_data('/home/cheng/data/nice_tag_data/synset_images_feat_7',
    #              '/home/cheng/data/nice_tag_data/synset_images_np')
    # exit()
    import random
    options = {}
    fi = "/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/fast_vector.txt.vec"
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(
        fi)
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size

    dataset = NICESynsetDataset(options,
                                '/home/chenguang/workhome/nice_tag_data/nice_synonym_data_v2/split_0/train_label.txt',
                                '/home/chenguang/workhome/nice_tag_data/synset_images_np100_tiny',
                                use_visual=True,
                                use_npy=True,
                                neg_sample_size=40, max_set_lengths=25)
    '''
    loader = dataloader.DataLoader(
        dataset, shuffle=True, batch_size=1000, collate_fn=word_collect_fn)
    prefetcher = DataPrefetcher(loader)

    data = prefetcher.next()
    idx = 0
    it = iter(loader)
    while True:
        try:
            a = next(it)
            print(idx)
            idx += 1
        except StopIteration:
            print('done')
            break
    idx = 0
    while True and data is not None:
        a, b, c = data
        # print(a.shape)
        # print(b.shape)
        # print(c.shape)
        data = prefetcher.next()
        idx += 1
        print(idx, '---', len(loader))

    '''
    saver = Hdf5Saver(dataset, 16, 5)
    saver.write_hdf5()
