#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 17:22
# @Author  : Pan, Jiahua
# @File    : loss_fn.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


def multilabel_categorical_cross_entropy(y_true, y_pred):
    """
    多标签分类交叉熵 https://kexue.fm/archives/7359
    log(1 + sum(exp(si)) for i in neg_classes) + log(1 + sum(exp(-si)) for i in pos_classes)
    y_true和y_pred的shape需要保持一致, y_true为硬标签, 元素只能是0或1, 1代表对应的类为目标类, 0代表对应的类为非目标类.
    y_pred不需要加入任何激活(softmax,sigmoid...)需要y_pred保持在实数域即可.
    :param y_true: ... * num_samples
    :type y_true: torch.Tensor
    :param y_pred: ... * num_samples
    :type y_pred: torch.Tenor
    :return: 多标签分类交叉熵损失, a 1 * 1 tensor
    :rtype: torch.Tensor
    """
    y_pred = (1 - 2 * y_true) * y_pred  # y_pred中正类位置*-1, 负类位置不变
    y_pred_neg = y_pred - y_true * 1e12  # y_pred正类位置抵消(变成极大的负数, 接下来做exp会变成0), 只剩下负类
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # y_pred负类位置抵消, 只剩下正类
    zeros = torch.zeros_like(y_pred[..., :1])  # 加上0, exp之后变成式中的 + 1
    y_pred_neg = torch.concat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.concat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


def global_pointer_cross_entropy(y_true, y_pred):
    """

    :param y_true: batch * num_heads * seq_len * seq_len
    :type y_true: torch.Tensor
    :param y_pred: batch * num_heads * seq_len * seq_len
    :type y_pred: torch.Tensor
    :return: 1 * 1 tensor
    :rtype: torch.Tensor
    """

    bh = y_true.shape[0] * y_true.shape[1]  # batch * num_heads
    y_true, y_pred = y_true.reshape((bh, -1)), y_pred.reshape((bh, -1))  # （batch * num_heads) * (seq_len * seq_len)
    return multilabel_categorical_cross_entropy(y_true, y_pred)




