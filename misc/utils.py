"""
.. module:: utils
    :synopsis: utility functions

.. moduleauthor:: Jiaming Shen
"""
from collections import defaultdict
from tqdm import tqdm
import mmap
import os
import logging
import torch
from gensim.models import KeyedVectors  # used to load word2vec
import hashlib
import itertools
import json
import shutil
import pickle
from loguru import logger


def check_tensor(tensor):
    """
    检测tensor是否可以防止上cuda上
    """
    if torch.cuda.is_available():
        return tensor.to('cuda')
    return tensor.to('cpu')


def check_parallel_model(model):
    """
    使用gpu
    """
    gpu_count = torch.cuda.device_count()
    print('now we use {} gpu'.format(gpu_count))
    if gpu_count == 1:
        return model.to('cuda')
    elif gpu_count > 1:
        return torch.nn.DataParallel(model)
    else:
        return model.to('cpu')


class Metrics:
    """ A metric class wrapping all metrics

    """

    def __init__(self):
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, metric_name, metric_value):
        """ Add metric value for the given metric name

        :param metric_name: metric name
        :type metric_name: str
        :param metric_value: metric value
        :type metric_value:
        :return: None
        :rtype: None
        """
        self.metrics[metric_name] = metric_value

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

def write_dict2vec(dic, filename, d=200):
    """保持embeding文件
    """
    words = len(dic.keys())
    with open(filename, 'w') as fp:
        fp.write('{} {}\n'.format(words, d))
        for k, v in dic.items():
            s = ' '.join([str(i) for i in v])
            sens = "{} {}\n".format(k, s)
            fp.write(sens)

class Results:
    """ A result class for saving results to file

    :param filename: name of result saving file
    :type filename: str
    """

    def __init__(self, filename):
        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save_metrics(self, hyperparams, metrics):
        """ Save model hyper-parameters and evaluation results to the file

        :param hyperparams: a dictionary of model hyper-parameters, keyed with the hyper-parameter names
        :type hyperparams: dict
        :param metrics: a Metrics object containg all model evaluation results
        :type metrics: Metrics
        :return: None
        :rtype: None
        """

        result = metrics.metrics  # a dict
        result["hash"] = self._hash(hyperparams)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum



loggers = {}

def set_logger(LOG_PATH):
    """
    设置logger
    """
    fmt = "{time} | {level} | {name} |p:{process} - t:{thread}| {module} : {function} : {line}| {message}"
    debug = logger.add(LOG_PATH, rotation="12:00", format=fmt, level="DEBUG",encoding='utf-8',)
    info = logger.add(LOG_PATH, rotation="12:00", format=fmt, level="INFO",encoding='utf-8',)
    loggers["debug"] = debug
    loggers["info"] = info

def load_embedding(fi, embed_name="word2vec"):
    """ Load pre-trained embedding from file

    :param fi: embedding file name
    :type fi: str
    :param embed_name: embedding format, currently only supports "word2vec" format embedding. c.f.: https://radimrehurek.com/gensim/models/keyedvectors.html
    :type embed_name: str
    :return:

        - embedding : embedding file
        - index2word: map from element index to element
        - word2index: map from element to element index
        - vocab_size: size of element pool
        - embed_dim: embedding dimension

    :rtype: (gensim.KeyedVectors, list, dict, int, int)
    """
    if embed_name == "word2vec":
        print(fi)
        embedding = KeyedVectors.load_word2vec_format(fi)
    else:
        # TODO: allow training embedding from scratch later
        print("[ERROR] Please specify the pre-trained embedding")
        exit(-1)

    vocab_size, embed_dim = embedding.vectors.shape
    index2word = ['PADDING_IDX'] + embedding.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    return embedding, index2word, word2index, vocab_size, embed_dim


def load_raw_data(fi):
    """ Load raw data from file

    :param fi: data file name
    :type fi: str
    :return: a list of raw data from file
    :rtype: list
    """

    raw_data_strings = []
    with open(fi, "r") as fin:
        for line in fin:
            raw_data_strings.append(line.strip())
    return raw_data_strings




def print_args(args, interested_args="all"):
    """ Print arguments in command line

    :param args: parsed command line argument
    :type args: Namespace
    :param interested_args: a list of interested argument names
    :type interested_args: list
    :return: None
    :rtype: None
    """
    print("\nParameters:")
    if interested_args == "all":
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
    else:
        for attr, value in sorted(args.__dict__.items()):
            if attr in interested_args:
                print("\t{}={}".format(attr.upper(), value))
    print('-' * 89)




def read_synset(fp):
    '''
    读取同义词集
    '''
    data = load_raw_data(fp)
    output = []
    for line in data:
        items = line.split('\t')
        items = [item.split('(')[0].replace(' ', '_') for item in items]
        output.append(items)
    return output


def write_synset(data):
    """
    data:[[w1,w2,...],....]
    """
    output = []
    for idx, line in enumerate(data):
        pre = 'c{}'.format(idx)
        # print(line)
        items = ','.join(['\"' + i + '\"' for i in line])
        out = "{} {{ {} }}".format(pre, items)
        output.append(out)
    return output


def save_evaluation_result(metric_cls, path, epoch='best'):
    result_path = os.path.join(path, '{}.result'.format(epoch))

    with open(result_path, 'w') as fp:
        for k, v in metric_cls.items():
            out = '{}\t{}\n'.format(k, v)
            fp.write(out)
    if isinstance(epoch, int):
        best_path = os.path.join(path, 'best.result')
        shutil.copy(result_path, best_path)


def save_pkl(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pkl(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)







    
