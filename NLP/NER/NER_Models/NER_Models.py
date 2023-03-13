#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/14 21:03
# @Author  : Pan, Jiahua
# @File    : NER_Models.py
# @Software: PyCharm

import torch
import torch.nn as nn
from .Position_Embeddings import RoPE
from transformers import BertModel, BertTokenizerFast, AutoModel, AutoTokenizer, BertConfig, AutoConfig
from transformers.models.bert import BertModel


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    # 排除padding
    # mask原本维度(batch, seq_len) --> (batch, num_heads, seq_len, seq_len)
    pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(*logits.shape)
    logits = logits * pad_mask - (1 - pad_mask) * 1e12
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


# NOTE: GLOBAL_POINTER, 全局指针网络，解决嵌套实体问题 https://spaces.ac.cn/archives/8373
#  实现思路：
#   1. bert_inputs: batch_size * seq_len * hidden_dim 经过 Linear层 --> batch * seq_len * (num_heads * head_dim * 2)
#   2. batch * seq_len * num_heads * (head_dim * 2) 直接对半reshape得到
#   qw 和 kw --> batch * seq_len * num_heads * head_dim, permute(0,2,1,3)之后与RoPE的位置编码按元素相乘
#   3. 通过RoPE得到位置编码 sin_pos和cos_pos: batch(1) * seq_len * head_dim
#   4. qw 和 kw 内积得到 logits --> batch * num_heads * seq_len * seq_len
#   5. 对logits排除下三角（不包含对角线), 进行scale
class GlobalPointer(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.linear = nn.Linear(hidden_dim, num_heads * head_dim * 2)
        self.pos = RoPE(head_dim)

    def forward(self, inputs, mask=None):
        """

        :param inputs: batch * seq_len * hidden_dim
        :type inputs: torch.Tensor
        :param mask: attention_mask, 排除padding
        :type mask: Union[None, torch.Tensor]
        :return: batch * num_heads * seq_len * seq_len
        :rtype: torch.Tensor
        """

        hidden_states = self.linear(inputs)  # hidden_states --> batch * seq_len * (num_heads * head_dim * 2)
        hidden_states = torch.split(hidden_states, (self.head_dim * 2),
                                    dim=-1)  # num_heads个张量 每个的维度是 batch * seq_len * (head_dim *2)
        hidden_states = torch.stack(hidden_states,
                                    dim=-2)  # hidden_states --> batch * seq_len * num_heads * (head_dim * 2)
        sin_pos, cos_pos = RoPE(self.head_dim)(hidden_states)  # sin_pos --> batch(1) * seq_len * head_dim
        qw, kw = hidden_states[..., :self.head_dim], hidden_states[..., self.head_dim:]
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], dim=-1)  # qw2 --> batch * seq_len * num_heads * head_dim
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], dim=-1)
        qw, qw2 = qw.permute(0, 2, 1, 3), qw2.permute(0, 2, 1, 3)
        kw, kw2 = kw.permute(0, 2, 1, 3), kw2.permute(0, 2, 1, 3)
        qw = qw * cos_pos + qw2 * sin_pos  # qw --> batch * num_heads * seq_len * head_dim
        kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bhnd, bhmd -> bhnm', (qw, kw))  # logits --> batch * num_heads * seq_len * seq_len
        return add_mask_tril(logits, mask=mask) / (self.head_dim ** 0.5)


class GlobalPointerNet(nn.Module):
    def __init__(self, num_categories, head_dim, model_path):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(model_path)
        self.bert_config = BertConfig.from_pretrained(model_path)
        self.global_pointer = GlobalPointer(num_heads=num_categories, head_dim=head_dim,
                                            hidden_dim=self.bert_config.hidden_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_output = self.bert_model(input_ids, token_type_ids, attention_mask).last_hidden_state
        logits = self.global_pointer(inputs=encoder_output, mask=attention_mask)
        return logits


if __name__ == '__main__':
    model_path = '../models/MacBERT'
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    gpnet = GlobalPointerNet(9, 64, model_path)
    test_sentence = ['这句话用来测试', '这是第二句话']
    tokenizer_output = tokenizer(test_sentence, max_length=128, padding='max_length', truncation=True,
                                 return_tensors='pt')
    logits = gpnet(**tokenizer_output)
    print(logits.shape)
