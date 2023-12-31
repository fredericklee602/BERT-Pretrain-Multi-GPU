
import os
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import BertModel, BertConfig
from model.BertForMaskedLM import BertForMaskedLM
from Config import Config
from transformers import DistilBertForMaskedLM
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, DistilBertTokenizer
from DataManager import DataManager


class Trainer(object):
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer) 

    def train(self, train_loader, valid_loader, local_rank):
        """
            預訓練模型
        """
        print('training start')
        # 初始化配置
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        device = torch.device("cuda", local_rank)
        print("cuda device:",device)

        # 初始化模型和優化器
        print('model loading')
        model = BertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        # model = DistilBertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")
        
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        if "epoch_" in self.config.train_start:
            opt_path = self.config.path_model_save + self.config.train_start
            opt_path = opt_path + "optimizer.pt"
            optimizer.load_state_dict(torch.load(opt_path))

        # 設定優化器配置
        num_training_steps = self.config.num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 分散式訓練
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
        print('start to train')
        model.train()
        progress_bar = tqdm(range(num_training_steps))
        loss_best = math.inf
        epoch_start = 0
        if "epoch_" in self.config.train_start:
            epoch_start = int(self.config.train_start.split("epoch_")[1])
        for epoch in range(epoch_start,epoch_start + self.config.num_epochs):
            # DDP：設置sampler的epoch，
            # DistributedSampler需要這個shuffle方式，
            # 通過維持各個Process之間的相同隨機Seed使不同Process能獲得同樣的shuffle效果。
            train_loader.sampler.set_epoch(epoch)
            for i, batch in enumerate(train_loader):
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                # 計算loss
                loss = outputs.loss
                loss = loss.mean()                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if i % 500 == 0:
                    print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_loader), loss.item()))
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                # 模型保存
                # DDP:
                # 1. save模型的时候，和DP模式一样，有一个需要注意的點：保存的是model.module而不是model。
                #    因為model其實是DDP model，參數是被`model=DDP(model)`包起來的。
                # 2. 只需要在Process 0上保存一次就行了，避免多次保存重複的東西。
                self.eval(valid_loader, model, epoch, device)
                model_save = model.module if torch.cuda.device_count() > 1 else model
                path = self.config.path_model_save + 'epoch_{}/'.format(epoch)
                model_save.save_pretrained(path)
                path = path + "optimizer.pt"
                torch.save(optimizer.state_dict(), path)


    def eval(self, eval_dataloader, model, epoch, device):
        print("=====  Validation Mode  =====")
        losses = []
        model.eval()
        
        input = []
        label = []
        pred = []
        for batch in eval_dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            loss = loss.unsqueeze(0)
            losses.append(loss)
            
            # 還原成token string   
            tmp_src = batch['input_ids'].cpu().numpy()
            tmp_label = batch['labels'].cpu().numpy()
            tmp_pred = torch.max(outputs.logits, -1)[1].cpu().numpy()
            for i in range(len(tmp_label)):
                line_l = tmp_label[i]
                line_l_split = [ x for x in line_l if x not in [0]]
                line_s = tmp_src[i]
                line_s_split = line_s[:len(line_l_split)]
                line_p = tmp_pred[i]
                line_p_split = line_p[:len(line_l_split)]
                tmp_s = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_s_split))
                tmp_lab = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_l_split))
                tmp_p = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_p_split))
                input.append(tmp_s)
                label.append(tmp_lab)
                pred.append(tmp_p)
        # 計算困惑度
        losses = torch.cat(losses)
        losses_avg = torch.mean(losses)
        perplexity = math.exp(losses_avg)
        print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))
        for i in range(10):
            print('-'*30)
            print('input: {}'.format(input[i]))
            print('label: {}'.format(label[i]))
            print('pred : {}'.format(pred[i]))
        
        return losses_avg
    
    def huge_data_train(self,local_rank,world_size):
        # 數據處理
        print('read data...')
        dm = DataManager(self.config)
        valid_loader = dm.get_dataset(mode='dev')

        """
            預訓練模型
        """
        print('training start')
        # 初始化配置
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        device = torch.device("cuda", local_rank)
        print("cuda device:",device)

        # 初始化模型和優化器
        print('model loading')
        model = BertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        # model = DistilBertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")
        file_list = os.listdir(self.config.data_path_prefix)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        if "epoch_" in self.config.train_start:
            opt_path = self.config.path_model_save + self.config.train_start
            opt_path = opt_path + "optimizer.pt"
            optimizer.load_state_dict(torch.load(opt_path))

        # 設定優化器配置
        num_training_steps = int(self.config.num_epochs * self.config.huge_data_file_data_length * len(file_list) / (self.config.batch_size*world_size))
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 分散式訓練
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
        print('start to train')
        model.train()
        progress_bar = tqdm(range(num_training_steps))
        loss_best = math.inf
        epoch_start = 0
        if "epoch_" in self.config.train_start:
            epoch_start = int(self.config.train_start.split("epoch_")[1])
        for epoch in range(epoch_start,epoch_start + self.config.num_epochs):
            file_completed = 0
            for shard in file_list:
                file_name = '/train_shard/'+shard
                train_loader = dm.data_process(file_name, self.tokenizer)
                # DDP：設置sampler的epoch，
                # DistributedSampler需要這個shuffle方式，
                # 通過維持各個Process之間的相同隨機Seed使不同Process能獲得同樣的shuffle效果。
                train_loader.sampler.set_epoch(epoch)
                for i, batch in enumerate(train_loader):
                    batch = {k:v.to(device) for k,v in batch.items()}
                    outputs = model(**batch)
                    # 計算loss
                    loss = outputs.loss
                    loss = loss.mean()                
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                    if (i+1) % 500 == 0:
                        print('epoch:{0}  iter:{1}/{2}  loss:{3} file_completed:{4}'.format(epoch, i, len(train_loader), loss.item(), file_completed))
                    if torch.distributed.get_rank() == 0 and (i+1)%2000==0:
                        # 模型保存
                        # DDP:
                        # 1. save模型的时候，和DP模式一样，有一个需要注意的點：保存的是model.module而不是model。
                        #    因為model其實是DDP model，參數是被`model=DDP(model)`包起來的。
                        # 2. 只需要在Process 0上保存一次就行了，避免多次保存重複的東西。
                        self.eval(valid_loader, model, epoch, device)
                        model_save = model.module if torch.cuda.device_count() > 1 else model
                        path = self.config.path_model_save + 'epoch_{}/'.format(epoch)
                        model_save.save_pretrained(path)
                        path = path + "optimizer.pt"
                        torch.save(optimizer.state_dict(), path)
                file_completed = file_completed + 1


if __name__ == '__main__':
    
    config = Config()
    train = Trainer()
    train(config)
    # load_lm()
