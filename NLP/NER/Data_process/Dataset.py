#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 15:56
# @Author  : Pan, Jiahua
# @File    : Dataset.py
# @Software: PyCharm
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_data(path, is_train):
    """
    加载CMeEE数据
    单条数据: [text, (start, end, label), (start, end, label), ...], 意味着text[start:end + 1]是类型为label的实体
    :param path: 数据路径
    :type path: str
    :param is_train: 是否为训练模式
    :type is_train: bool
    :return: 加载好的数据, 遵循 [text, (start, end, label), (start, end, label), ...] 格式
    :rtype: list, list
    """
    data_list = []
    label_set = set()
    with open(path, encoding='utf-8') as f:
        file = json.load(f)
        for data in file:
            data_list.append([data['text']])
            for ent in data['entities']:
                start, end, label = ent['start_idx'], ent['end_idx'], ent['type']
                if start <= end:
                    data_list[-1].append((start, end, label))
                label_set.add(label)
    label_categories = list(label_set)
    label_categories.sort()
    if is_train:
        return data_list, label_categories
    else:
        return data_list


class EEDataset(Dataset):
    def __init__(self, data, tokenizer, num_categories, categories2id, max_len=256):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.num_categories = num_categories
        self.max_len = max_len
        self.categories2id = categories2id

    def __getitem__(self, index):
        # 单条数据格式[text, (start, end, label), (start, end, label), ...]
        d = self.data[index]
        multi_head_labels = torch.zeros((self.num_categories, self.max_len, self.max_len))
        context = self.tokenizer(d[0], max_length=self.max_len, truncation=True,
                                 padding='max_length', return_tensors='pt', return_offsets_mapping=True)
        token2char_mapping = self.tokenizer(d[0], max_length=self.max_len, truncation=True,
                                            padding='max_length', return_offsets_mapping=True)['offset_mapping']
        input_ids = context['input_ids'].squeeze(0)
        token_type_ids = context['token_type_ids'].squeeze(0)
        attention_mask = context['attention_mask'].squeeze(0)
        start_map = {j[0]: i for i, j in enumerate(token2char_mapping) if j != (0, 0)}
        end_map = {j[0]: i for i, j in enumerate(token2char_mapping) if j != (0, 0)}
        for entities in d[1:]:
            start, end, label = entities[0], entities[1], entities[2]
            if start < self.max_len and end < self.max_len and start in start_map and end in end_map:
                start = start_map[start]
                end = end_map[end]
                multi_head_labels[self.categories2id[label], start, end] = 1
        return input_ids, token_type_ids, attention_mask, multi_head_labels

    def __len__(self):
        return len(self.data)


def get_dataloader(file_path, batch_size, is_train, tokenizer, num_categories=None, categories2id=None):
    if is_train:
        data, label_categories = load_data(file_path, is_train=is_train)
        num_cat = len(label_categories)
        cat2id = {cat: i for i, cat in enumerate(label_categories)}
        id2cat = {i: cat for i, cat in enumerate(label_categories)}
        train_dataset = EEDataset(data=data, tokenizer=tokenizer, num_categories=num_cat, categories2id=cat2id)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, num_cat, cat2id, id2cat
    else:
        data = load_data(file_path, is_train=is_train)
        valid_dataset = EEDataset(data=data, tokenizer=tokenizer, num_categories=num_categories,
                                  categories2id=categories2id)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
        return valid_dataloader
