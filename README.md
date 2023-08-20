# MLM pretrain BERT Model
## 專案說明

* 使用台灣繁體中文資料集訓練https://huggingface.co/datasets/yentinglin/zh_TW_c4
* 多片GPU並行訓練。
* Mask Language Model 預訓練BERT模型。

## Datasets
* `datasets/train.txt`：訓練用文本，可於write_data.ipynb做生成。
* `datasets/dev.txt`：每個訓練epoch結束後，會用dev.txt做驗證。
* `datasets/test.txt`：測試用資料。

## Environment
* Ubuntu20.04
* CUDA Version: 11.7
* GeForce RTX 3090 * 4
* python的版本為: 3.10.11
```
torch==1.13.1
transformers==4.29.2
```
* 會需要用到NVIDIA/apex，得將apex git clone再安裝才行。
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ python setup.py install
```
## 程式碼範例來源DEBUG
* 因為該開源碼我測試只能用單片GPU訓練，所以改動了部分程式。
* command 執行改動。
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29301' --use_env main.py
```
* 初始化GPU process設置改動，解決卡頓的問題。
```
torch.distributed.init_process_group(backend="nccl")
```
* 多process執行時將各個local_rank號碼寫入model，解決所有process只將model載入到GPU 0問題。
```
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
```

## Get Start

### 單卡模式(測試)

修改`Config.py`文件中的`self.path_model_predict`。

選擇想要的第n個epoch訓練的model再運行。
* 如要訓練到第9個epoch的Model參數，則輸入`self.path_model_predict = os.path.join(self.path_model_save, 'epoch_9')`
```
python main.py test
```

### 多卡模式（訓練）
如果你足夠幸運，擁有了多張GPU卡，那麼恭喜你，你可以進入起飛模式。🚀🚀

修改`Config.py`文件中的 `self.num_epochs, self.batch_size, self.sen_max_length` ，再運行。

* 設置訓練10個epoch，則輸入`self.num_epochs = 10`
* 設置 BERT長度(<=512)，如設置最長則輸入`self.sen_max_length = 512`
* 設置 batch_size大小(依照可容納size設置)，如設置16則輸入`self.batch_size = 16`
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29301' --use_env main.py train
```

* 使用torch的`nn.parallel.DistributedDataParallel`模塊進行多卡訓練。
* <font color=#009393>`master_port`：master節點的port號，在不同的節點上master_addr和master_port的設置是一樣的，用來進行通信，port我設置'29301'。</font>
* <font color=#009393>`nproc_per_node`：一個節點中顯卡的數量，我有4片GPU，所以設置4。 </font>

### 超大資料量讀取(訓練) [2023/08/17改動]
如果有資料量大到CPU RAM無法讀取的情況，請先將檔案分割寫入到路徑`./datasets/train_shard`

修改`Config.py`文件中的`self.huge_data_file_data_length`，每個檔案的資料有多少筆，則輸入多少。我分割成每個檔案160000筆，則輸入160000。
* 本人使用的資料量有百萬以上，https://huggingface.co/datasets/yentinglin/zh_TW_c4 的資料總共有500萬筆數、5 Billoin的tokens。
* 使用`torch.distributed.launch`執行有個優點及缺點。
- 優點：多process可以快速將DistributedSampler(tokenized_datasets)完成
- 缺點：會多次讀取相同檔案再整理進DistributedSampler()。
* 假設有4片GPU，`world_size=4`，會導致相同資料檔案同時被重複讀取四次。本人CPU RAM 有128GB依然不夠。
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29301' --use_env main.py huge_train
```

* 在`Trainer.py`新增`def huge_data_train()`，可以看出在訓練過程會讀取下個檔案再轉成新的DataLoader形式，之前是直接所有要訓練資料轉成DataLoader。
```
# Trainer.py
def huge_data_train(self,local_rank,world_size):
    ...
    for epoch in range(self.config.num_epochs):
      for shard in file_list:
        file_name = '/train_shard/'+shard
        train_loader = dm.data_process(file_name, self.tokenizer)
    ...
```
* 於Training過程反覆讀取新檔案，再創建新的DataLoader會有個問題。
* 優化器的 `learning rate scheduler` 參數 `training steps` 得重新精算。
* 所以得於`Config.py`文件中的 `self.huge_data_file_data_length` 輸入每個檔案的資料有多少筆。
```
num_training_steps = int(self.config.num_epochs * self.config.huge_data_file_data_length * len(file_list) / (self.config.batch_size*world_size))
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=self.config.num_warmup_steps,
    num_training_steps=num_training_steps
)
```
## training
使用交叉熵（cross-entropy）作為損失函數，困惑度（perplexity）和Loss作為評價指標來進行訓練。

## test
結果保存在`dataset/output/pred_data.csv`，分別包含三列：
- `src`表示原始輸入
- `pred`表示模型預測
- `mask`表示模型輸入（帶有mask和pad等token）

## 範例

```
src:  [CLS] art education and first professional work [SEP]
pred: [CLS] art education and first class work [SEP]
mask: [CLS] art education and first [MASK] work [SEP] [PAD] [PAD] [PAD] ...
```


# Reference

【Bert】[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)

【transformers】[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

【datasets】[https://huggingface.co/datasets/yentinglin/zh_TW_c4](https://huggingface.co/datasets/yentinglin/zh_TW_c4)




