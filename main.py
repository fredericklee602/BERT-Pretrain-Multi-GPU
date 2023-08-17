

import os
import datetime
import numpy as np
import torch
import logging
from Config import Config
from DataManager import DataManager
from Trainer import Trainer
from Predictor import Predictor
import sys


if __name__ == '__main__':

    config = Config()
    config.mode = sys.argv[1]
    start_time = datetime.datetime.now()
    # if config.mode in ['train', 'test']:
    print ("Currently using [%s] mode" % (sys.argv[1]))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # Set Seed，保證结果每次結果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # 模式
    if config.mode == 'train':
        # 數據處理
        print('read data...')
        dm = DataManager(config)
        local_rank = torch.distributed.get_rank()
        print("local_rank:",local_rank)
        torch.cuda.set_device(local_rank)
        # 獲取數據
        print('data process...')
        train_loader = dm.get_dataset(mode='train')
        valid_loader = dm.get_dataset(mode='dev')
        # 訓練
        trainer = Trainer(config)
        trainer.train(train_loader, valid_loader, local_rank)
        if local_rank==0:
            end_time = datetime.datetime.now()
            print("The train mode time spent:", end_time - start_time)
    elif config.mode == 'huge_train':
        torch.distributed.init_process_group(backend="nccl")
        print('world_size:', torch.distributed.get_world_size())
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        local_rank = torch.distributed.get_rank()
        print("local_rank:",local_rank)
        torch.cuda.set_device(local_rank)
        # 訓練
        trainer = Trainer(config)
        world_size = torch.distributed.get_world_size()
        trainer.huge_data_train(local_rank,world_size)
        if local_rank==0:
            end_time = datetime.datetime.now()
            print("The train mode time spent:", end_time - start_time)
    elif config.mode == 'test':
        # 數據處理
        print('read data...')
        dm = DataManager(config)
        # 測試
        test_loader = dm.get_dataset(mode='test', sampler=False)
        predictor = Predictor(config)
        predictor.predict(test_loader)
        end_time = datetime.datetime.now()
        print("The test mode time spent:", end_time - start_time)
    else:
        print("no task going on!")
        print("Train mode: python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=port --use_env main.py train")
        print("Test mode: python main.py test")
        