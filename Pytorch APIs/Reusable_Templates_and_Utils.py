#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 21:00
# @Author  : Pan,Jiahua
# @File    : Reusable_Templates_and_Utils.py
# @Software: PyCharm

"""
This File is for saving all kinds of reusable templates and utils
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# TODO: 多任务学习


# TODO: R-Drop(直接用作损失函数) (DONE)

class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """

    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, logits1, logits2, target, kl_weight=1.):
        """

        :param logits1: One output of the classification model.
        :type logits1: torch.Tensor
        :param logits2: Another output of the classification model.
        :type logits2: torch.Tensor
        :param target: The target labels.
        :type target: torch.Tensor
        :param kl_weight: The weight for `kl_loss`.
        :type kl_weight: float
        :return: Losses with the size of the batch size.
        :rtype: torch.Tensor
        """
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss.mean()


# TODO: 各种随机种子 (DONE)
def seed_everything(seed):
    """
    For all kinds of seed, including numpy and pytorch and torch.cuda
    :param seed: A random seed number
    :type seed: int
    :return: None
    :rtype: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# TODO: BERT类模型带 Mean pool(DONE)
class TextModel(nn.Module):
    def __init__(self, model_name_or_path, hidden_dim=128):
        super(TextModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_name_or_path)
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.linear = nn.Linear(self.bert_config.hidden_size, hidden_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.mean_pooling(self.bert_model(input_ids, token_type_ids, attention_mask).last_hidden_state,
                                    attention_mask)
        outputs = self.linear(outputs)
        return outputs

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        output_vectors = []
        # [B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        # [B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        # 限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # 得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
