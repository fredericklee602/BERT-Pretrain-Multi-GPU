
import os
import random




class Config(object):
    
    def __init__(self):
        
        self.mode = ''
        # GPU配置
        self.cuda_visible_devices = '0,1,2,3'                           # 可用的GPU
        self.device = 'cuda'                                      # master GPU
        # self.port = str(random.randint(10000,60000))                # 多卡訓練Process間通信Port
        # self.init_method = 'tcp://localhost:' + self.port           # 多卡訓練的通訊地址
        # self.world_size = 1                                        # Process 數，默認為1
        
        # 訓練配置
        self.whole_words_mask = True                                # 使用是否whole words masking機制
        self.num_epochs = 1                                       # 迭代次數
        self.batch_size = 16                                       # 每個批次的大小
        self.learning_rate = 3e-5                                   # 學習率
        self.num_warmup_steps = 0.1                                 # warm up步數
        self.sen_max_length = 512                                   # 句子最長長度
        self.padding = True                                         # 是否對輸入進行padding

        # 模型及路徑配置
        self.initial_pretrain_model = 'bert-base-chinese'           # 加載的預訓練分詞器checkpoint，默認為英文。若要選擇中文，替换成 bert-base-chinese
        self.initial_pretrain_tokenizer = 'bert-base-chinese'       # 加載的預訓練模型checkpoint，默認為英文。若要選擇中文，替换成 bert-base-chinese
        self.path_model_save = './checkpoint/bert/'                      # 模型保存路徑
        self.path_datasets = './datasets/'                          # 數據集
        self.path_log = './logs/'
        self.path_model_predict = os.path.join(self.path_model_save, 'epoch_9')
