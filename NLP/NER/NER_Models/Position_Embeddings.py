#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 17:27
# @Author  : Pan, Jiahua
# @File    : Position_Embeddings.py
# @Software: PyCharm

import torch
import torch.nn as nn
from transformers import BertModel


# NOTE: 定义三角函数式位置Embedding, 论文《Attention is all you need》提出的一个显式解
#   pk,2i = sin(k/10000^(2i/d)), pk,2i+1 = cos(k/10000^(2i/d))
#   三角函数式位置编码的特点是有显式的生成规律,因此可以期望于它有一定的外推性。另外一个使用它的理由是：由于sin(α+β)=sinαcosβ+cosαsinβ
#   以及cos(α+β)=cosαcosβ−sinαsinβ, 这表明位置α+β的向量可以表示成位置α和位置β的向量组合,这提供了表达相对位置信息的可能性
#   实现思路
#   1. position_ids 是0到(seq_len-1)的序列, 在第0维加上一维, 方便在之后的运算里与inputs的batch_size维度对齐 shape: 1, seq_len
#   2. indices是0到(dim//2)的序列, 用来计算三角函数式位置 pk,2i 和  pk,2i+1, shape均为: dim//2
#   3. position_ids与indices相乘并送入sin与cos函数得到位置embedding,
#   stack起来再reshape得到与inputs的shape一致的embedding. shape: 1 * seq_len * dim
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, mode='zero'):
        """

        :param dim: position embedding 的维度(常见的有512,768. GlobalPointer里是head dim 128)
        :type dim: int
        :param mode: 融合位置编码的形式 `add`代表加性, `mul`代表乘性, `zero`代表不做改变
        :type mode: str
        """
        super().__init__()
        assert mode in ['add', 'mul', 'zero'], "mode must be one of 'add', 'mul, 'zero'"
        assert dim % 2 == 0, 'dim must be an even number'
        self.mode = mode
        self.dim = dim
        self.base = base

    def forward(self, inputs):
        """
        inputs第二维是seq_len
        :param inputs: batch * seq_len * dim
        :type inputs: torch.Tensor
        :return: ... * seq_len * dim
        :rtype: torch.Tensor
        """
        seq_len = inputs.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.float).unsqueeze(0)  # 为了和inputs保持一致,在前面加1个维度与batch维度匹配
        indices = torch.arange(self.dim // 2, dtype=torch.float)
        indices = torch.pow(self.base, -2 * indices / self.dim)
        embedding = torch.einsum('...,d->...d', position_ids, indices)
        embedding = torch.stack((torch.sin(embedding), torch.cos(embedding)), dim=-1)
        # embedding shape --> ... * (d/2) * 2
        embedding = torch.reshape(embedding, (-1, seq_len, self.dim))
        # embedding shape --> ... * d 与inputs保持了一致
        if self.mode == 'add':
            return inputs + embedding.to(inputs.device)
        elif self.mode == 'mul':
            return inputs * (embedding + 1.0).to(inputs.device)
        elif self.mode == 'zero':
            return embedding.to(inputs.device)


# NOTE: 配合GlobalPointer旋转式位置编码(Rotary Position Embedding,RoPE)以一种绝对位置编码的方式实现相对位置编码
#   https://spaces.ac.cn/archives/8265
#   实现思路：
#   1. input shape --> batch * seq_len * num_heads * head_dim,
#   我们需要seq_len 和 head_dim作为最后两维与接下来的sin,cos矩阵进行点乘
#   2. 用 SinusoidalPositionEmbedding 得到 batch(1) * seq_len * head_dim, 通过切变分出sin 和 cos, 再repeat_interleave 得到
#   shape 均为 batch(1) * seq_len * head_dim 的 sin 和 cos RoPE张量
class RoPE(nn.Module):
    """
    配合GlobalPointer旋转式位置编码(Rotary Position Embedding,RoPE)以一种绝对位置编码的方式实现相对位置编码
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        """
        给inputs 附加旋转式位置编码信息
        :param inputs: batch * seq_len * ... * dim
        :type inputs: torch.Tensor
        :return: batch * seq_len * ... * dim
        :rtype: torch.Tensor
        """
        # seq_len = inputs.shape[1]
        pos_emb = SinusoidalPositionEmbedding(self.dim)(inputs)
        sin_pos, cos_pos = pos_emb[..., ::2], pos_emb[..., 1::2]  # shape --> batch(1) * seq_len * dim // 2
        sin_pos = torch.repeat_interleave(sin_pos, 2, dim=-1)
        cos_pos = torch.repeat_interleave(cos_pos, 2, dim=-1)
        return sin_pos, cos_pos


if __name__ == '__main__':
    inputs = torch.rand((16, 150, 768))
    sinusoidal_position_emb = SinusoidalPositionEmbedding(64)
    rope = RoPE(64)
    emb_sinu = sinusoidal_position_emb(inputs)
    emb_rope = rope(inputs)
    print(emb_rope[0].shape, emb_rope[1].shape)
    print(emb_sinu.shape)
