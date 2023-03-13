#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 16:13
# @Author  : Pan, Jiahua
# @File    : Train.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import time
from Data_process.Dataset import load_data, EEDataset, get_dataloader
from loss_function.loss_fn import global_pointer_cross_entropy
from Metrics.metrics import get_evaluate_fpr, global_pointer_f1_score
from NER_Models.NER_Models import GlobalPointerNet
from utils.utils import seed_everything
from transformers import BertTokenizerFast, get_cosine_schedule_with_warmup, AdamW, BertConfig


def evaluate(model, valid_dataloader, device):
    model.to(device)
    model.eval()
    total_f1_, total_precision_, total_recall_ = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, (ids, tpe, att, label) in tqdm((enumerate(valid_dataloader))):
            y_pred = model(ids.to(device), tpe.to(device), att.to(device))
            f1, precision, recall = get_evaluate_fpr(label, y_pred)
            total_f1_ += f1
            total_precision_ += precision
            total_recall_ += recall
        avg_f1 = total_f1_ / (len(valid_dataloader))
        avg_precision = total_precision_ / (len(valid_dataloader))
        avg_recall = total_recall_ / (len(valid_dataloader))
        print(f"EVAL_F1:{avg_f1}\tEVAL_Precision:{avg_precision}\tEVAL_Recall:{avg_recall}\t")
    return avg_f1


def train_and_eval(model, train_dataloader, valid_dataloader, optimizer, scheduler, device, epochs, model_save_path):
    model.to(device)
    best_f1 = 0.0
    start_time = time.perf_counter()
    for i in tqdm(range(epochs), desc="EPOCHS"):
        """训练模型"""
        model.train()
        print(f"***** Running training epoch {i + 1} *****")
        train_loss_sum, total_sample_f1 = 0.0, 0.0
        for idx, (ids, tpe, att, label) in enumerate(tqdm(train_dataloader)):
            ids, tpe, att, label = ids.to(device), tpe.to(device), att.to(device), label.to(device)
            pred = model(ids, tpe, att)
            loss = global_pointer_cross_entropy(label, pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # 学习率变化
            sample_f1 = global_pointer_f1_score(label, pred)
            total_sample_f1 += sample_f1.item()
            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_dataloader) // 5) == 0:
                print(f"Epoch {i + 1} | Step {idx + 1}/{len(train_dataloader)}")
                print(f"Loss: {train_loss_sum / (idx + 1)} | Sample F1: {total_sample_f1 / (idx + 1)}")
                print(f"Learning rate = {optimizer.state_dict()['param_groups'][0]['lr']}")
        avg_f1 = evaluate(model, valid_dataloader, device)
        if avg_f1 > best_f1:
            torch.save(model.state_dict(), f'{model_save_path}/best_NERModel.pth')
            best_f1 = avg_f1
        print(f"current f1 score is {avg_f1}, best f1 score is {best_f1}")
    end_time = time.perf_counter()
    print(f"The training is over, the overall training takes {end_time - start_time} sec")


if __name__ == '__main__':
    config_file = open('./Config/Config.yaml', 'r')
    Config = yaml.safe_load(config_file)
    # 配置参数
    DATA_PATH = Config['data_path']
    Train_dataset = Config['train_dataset']
    Valid_dataset = Config['valid_dataset']
    Model_root_path = Config['model_root_path']
    Model_name = Config['model_name']
    Model_full_path = Model_root_path + Model_name
    Seed = Config['seed']
    Epochs = Config['epochs']
    Batch_size = Config['batch_size']
    Warmup_ratio = Config['warmup_ratio']
    Head_dim = Config['global_pointer_head_dim']
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(Seed)
    # 配置模型以及训练器等
    Tokenizer = BertTokenizerFast.from_pretrained(Model_full_path)
    Train_dataloader, num_cat, cat2id, id2cat = get_dataloader(file_path=DATA_PATH + Train_dataset,
                                                               batch_size=Batch_size, is_train=True,
                                                               tokenizer=Tokenizer)
    Valid_dataloader = get_dataloader(file_path=DATA_PATH + Valid_dataset, batch_size=Batch_size, is_train=False,
                                      tokenizer=Tokenizer, num_categories=num_cat, categories2id=cat2id)
    Ner_model = GlobalPointerNet(num_categories=num_cat, head_dim=Head_dim, model_path=Model_full_path)
    optimizer = AdamW(Ner_model.parameters(), lr=5e-5, weight_decay=1e-4)
    total_steps = Epochs * len(Train_dataloader)
    warmup_steps = Warmup_ratio * total_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # 训练
    train_and_eval(model=Ner_model, train_dataloader=Train_dataloader, valid_dataloader=Valid_dataloader,
                   optimizer=optimizer, scheduler=scheduler, device=Device, epochs=Epochs,
                   model_save_path=Model_root_path)
