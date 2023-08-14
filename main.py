

import os
import time
import numpy as np
import torch
import logging
from Config import Config
from DataManager import DataManager
from Trainer import Trainer
from Predictor import Predictor



if __name__ == '__main__':


    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # Set Seed，保證结果每次結果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # 數據處理
    print('read data...')
    dm = DataManager(config)

    # 模式
    if config.mode == 'train':
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
    elif config.mode == 'test':
        # 測試
        test_loader = dm.get_dataset(mode='test', sampler=False)
        predictor = Predictor(config)
        predictor.predict(test_loader)
    else:
        print("no task going on!")
        print("you can use one of the following lists to replace the valible of Config.py. ['train', 'test', 'valid'] !")
        