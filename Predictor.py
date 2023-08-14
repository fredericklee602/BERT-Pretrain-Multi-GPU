
import os
from posixpath import sep
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from apex import amp
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup,AutoTokenizer
# from transformers import BertTokenizer, BertConfig, AutoConfig, BertForMaskedLM, DistilBertForMaskedLM, DistilBertTokenizer, AutoTokenizer
from model.BertForMaskedLM import BertForMaskedLM
from sklearn import metrics
from Config import Config



class Predictor(object):
    
    def __init__(self, config):
        self.config = config
        # self.test_loader = test_loader
        self.device = torch.device(self.config.device)
        # 加載模型
        self.load_tokenizer()
        self.load_model()
    
    
    def load_tokenizer(self):
        """
        讀取分詞器
        """
        print('loading tokenizer config ...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)
    
    
    def load_model(self):
        """
        加載模型及初始化模型參數
        """
        print('loading model...%s' %self.config.path_model_predict)
        self.model = BertForMaskedLM.from_pretrained(self.config.path_model_predict)
        # 將模型加載到CPU/GPU
        self.model.to(self.device)
        self.model.eval()
    
    
    def predict(self, test_loader):
        """
        預測
        """
        print('predict start')        
        # 初始化指標計算
        src = []
        label = []
        pred = []
        input = []
        print("Batch Length:{0}".format(len(test_loader)))
        for i, batch in enumerate(test_loader):
            # 推斷
            batch = {k:v.to(self.config.device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                outputs_pred = outputs.logits
            if i%100==0:
                print("Batch No.{0} completed".format(i))
            # 還原成token string    
            tmp_src = batch['input_ids'].cpu().numpy()
            tmp_label = batch['labels'].cpu().numpy()
            tmp_pred = torch.max(outputs_pred, -1)[1].cpu().numpy()
            for j in range(len(tmp_label)):
                line_s = tmp_src[j]
                line_l = tmp_label[j]
                line_l_split = [ x for x in line_l if x not in [0]]
                line_p = tmp_pred[j]
                line_p_split = line_p[:len(line_l_split)]
                tmp_s = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_s))
                tmp_s = tmp_s.replace('[PAD] ','')
                tmp_lab = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_l_split))
                tmp_p = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_p_split))
                input.append(tmp_s)
                label.append(tmp_lab)
                pred.append(tmp_p)
            
        # 計算指標
        total = 0
        count = 0
        for k,(s,t) in enumerate(zip(label, pred)):
            if '[MASK]' in input[k]:
                total += 1
                if s==t:
                    count += 1
        acc = count/max(1, total)
        print('\nTask: acc=',acc)
        
        # 保存
        # Task 1
        data = {'src':label, 'pred':pred, 'mask':input}
        data = pd.DataFrame(data)
        path = os.path.join(self.config.path_datasets, 'output')
        if not os.path.exists(path):
            os.mkdir(path)
        path_output = os.path.join(path, 'pred_data.csv')
        data.to_csv(path_output, sep='\t', index=False)
        print('Task 1: predict result save: {}'.format(path_output))

        
        