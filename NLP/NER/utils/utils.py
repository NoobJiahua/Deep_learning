#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 10:50
# @Author  : Pan, Jiahua
# @File    : utils.py
# @Software: PyCharm

import os
import random
import numpy as np
import torch


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
