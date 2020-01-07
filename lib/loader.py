from __future__ import division, print_function
import warnings
import numpy as np
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
import csv
import pickle
import os
import os.path as osp
import logging
import torch
from torch.utils.data import Dataset, DataLoader


class CsvDataset:
    def __init__(self, filename='data.csv', line_count=1721578, data_count=0):
        pkl_name = '{}.pkl'.format('.'.join(filename.split('.')[:-1]))

        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as fi:
                self.data = pickle.load(fi)
                if data_count == 0:
                    self.count = len(self.data['UpdateTime'])
                else:
                    self.count = data_count
                    self.data = self.data[:data_count]

        else:
            self.data = {}
            self.count = 0

            with open(filename, 'r') as csvfile:
                line = csvfile.readline().rstrip('\n')
                keys = line.split(',')
                for k in keys:
                    self.data[k] = []

            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                with tqdm(total=(line_count-1 if data_count == 0 else data_count)) as pbar:
                    for row in reader:
                        for k, v in row.items():
                            # if re.match('[0-2][0-9]:[0-5][0-9]:[0-5][0-9]', v):
                            if k == 'UpdateTime':
                                t = datetime.strptime(v, '%H:%M:%S')
                                self.data[k].append(t.hour * 3600 + t.minute * 60 + t.second)
                            else:
                                self.data[k].append(eval(v))

                        self.count += 1
                        pbar.update()
                        if self.count == data_count:
                            break

            # self.data['label'] = []

            if data_count == 0:
                with open(pkl_name, 'wb') as fo:
                    pickle.dump(self.data, fo)

        for k, v in self.data.items():
            setattr(self, k, v)

    def __getitem__(self, index):
        item = {}
        for k, v in self.data:
            item[k] = v[index]
        return item

    def __len__(self):
        return self.count


class MemmapDataset():
    def __init__(self, root_dir='data', meta_filename='meta.pkl'):
        with open(osp.join(root_dir, meta_filename), 'rb') as fi:
            meta = pickle.load(fi)
        self.data = {}
        self.count = meta['data_count']
        for key, (mmap_filename, dtype) in meta['memmaps'].items():
            self.data[key] = np.memmap(osp.join(root_dir, mmap_filename),
                                       dtype=dtype, mode='r', shape=(self.count,))
        logging.debug('Data loaded successfully')

    def __getitem__(self, index):
        if index >= self.count or index < -self.count:
            raise IndexError('dataset index out of range')
        item = {}
        for k, v in self.data.items():
            item[k] = v[index]
        return item

    def __len__(self):
        return self.count


class ProcessedCsvDataset():
    def __init__(self, root_dir='data', normalized=True):
        train = pd.read_csv('data/Train.csv')
        test = pd.read_csv('data/Test.csv')
        feature_name = train.columns.values[:-1].tolist()
        train_feature_raw = train[feature_name]
        train_label = train['label']
        test_feature_raw = test[feature_name]
        test_label = test['label']

        # Normalize
        if normalized:
            mean = train_feature_raw.mean()
            std = train_feature_raw.std()
            self.train_feature = ((train_feature_raw - mean) / std).values
            self.test_feature = ((test_feature_raw - mean) / std).values
        else:
            self.train_feature = train_feature_raw.values
            self.test_feature = test_feature_raw.values

        self.train_label = train_label.values
        self.test_label = test_label.values

        self.num_features = self.train_feature.shape[1]

    def make_val_from_test(self, fraction=0.1):
        N = self.test_feature.shape[0]
        if hasattr(self, 'val_index'):
            warnings.warn("Re-sampling validation set")
        self.val_index = np.random.choice(N, int(N*fraction), replace=False)
        self.val_feature = self.test_feature[self.val_index]
        self.val_label = self.test_label[self.val_index]

        return self.val_index


class OneDataset(Dataset):
    def __init__(self, features):
        self.data = features
        self.num_features = self.data.shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class One2OneDataset(Dataset):
    def __init__(self, features, labels):
        self.data = features
        self.labels = labels
        assert self.data.shape[0] == self.labels.shape[0], "Shapes do not match"
        self.num_features = self.data.shape[1]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class ManyDataset(Dataset):
    def __init__(self, features, len_sequence=10):
        self.data = features
        self.len_sequence = len_sequence
        self.num_data, self.num_features = features.shape
        self.num_data = self.num_data - len_sequence + 1

    def __getitem__(self, index):
        return self.data[index:index+self.len_sequence]

    def __len__(self):
        return self.num_data


class Many2OneDataset(Dataset):
    def __init__(self, features, labels, len_sequence=10):
        self.data = features
        self.labels = labels
        self.len_sequence = len_sequence
        self.num_data, self.num_features = features.shape
        self.num_data = self.num_data - len_sequence + 1

    def __getitem__(self, index):
        return self.data[index:index+self.len_sequence], self.labels[index]

    def __len__(self):
        return self.num_data


def get_loader(dataset, batch_size=1, shuffle=True, seq_first=False, **kwargs):
    def collate(batch):
        b = list(zip(*batch))
        x, y = b
        y = torch.from_numpy(np.stack(y, 0))
        x = torch.from_numpy(np.stack(x, 1))
        if y is None:
            return x
        return x, y

    if not seq_first or \
            isinstance(dataset, OneDataset) or isinstance(dataset, One2OneDataset):
        collate = None

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate, **kwargs)


if __name__ == '__main__':
    dataset = Dataset()
    for i in range(10):
        print(dataset[i])

