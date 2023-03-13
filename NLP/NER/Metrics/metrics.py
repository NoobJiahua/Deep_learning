#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 11:02
# @Author  : Pan, Jiahua
# @File    : metrics.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn

nn.CrossEntropyLoss()


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)  # TP + FP(我们预测值中大于0的个数)
    numerator = torch.sum(y_true * y_pred)  # TP
    denominator = torch.sum(y_true + y_pred)  # TP + FP + TP + FN
    return 2 * numerator / denominator


def get_evaluate_fpr(y_true, y_pred):
    """
    因为global_pointer等多头标注的方法的设计,我们可以直接计算实体级别的f1_score, precision, recall
    y_true由0和1组成, 0代表负类, 1代表正类. y_pred中的元素皆在实数域内, 只要上三角的元素大于0,我们就认为模型预测了一个实体
    f1 = 2/(precision^-1 + recall^-1) = 2 * TP/(TP+FP+TP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    :param y_true: 真实标签
    :type y_true: torch.Tensor
    :param y_pred: 预测值
    :type y_pred: torch.Tensor
    :return: 实体级别 f1 score
    :rtype: torch.Tensor
    """
    y_pred = y_pred.cpu().numpy()  # batch * num_heads * seq_len * seq_len
    y_true = y_true.cpu().numpy()
    pred_set = set()
    true_set = set()
    for b, h, start, end in zip(*np.where(y_pred > 0)):
        pred_set.add((b, h, start, end))
    for b, h, start, end in zip(*np.where(y_true > 0)):
        true_set.add((b, h, start, end))

    X = len(pred_set & true_set)  # TP
    Y = len(pred_set)  # TP + FP
    Z = len(true_set)  # TP + FN
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall
