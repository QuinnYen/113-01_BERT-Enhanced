"""
BERT 情緒分析模型
用於文本情緒分類（正面/負面）
"""

# === 導入所需套件 ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 日誌
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 關閉 oneDNN 客製化操作

import time
import json
import torch
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def set_seed(seed):
    """設定所有隨機種子以確保實驗的可重複性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 控制 PyTorch 的警告級別
warnings.filterwarnings('ignore')
# 設定全局設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    # 設定 CUDA 相關參數
    torch.cuda.empty_cache()
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體使用：{torch.cuda.memory_allocated()/1024**2:.2f} MB")
else:
    print("使用 CPU 進行訓練")

# === 設定日誌 ===
def setup_logging(output_dir):
    """設定日誌"""
    log_file = os.path.join(output_dir, 'logs', 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

# === 設定參數 ===
def get_parameters():
    return {
        # 基本參數
        "num_class": 2,                                  # 分類類別數量（2表示二分類：正面/負面）
        "time": str(datetime.now()).replace(" ", "_"),   # 當前時間，用於命名
        "seed": 1111,                                    # 隨機種子，確保實驗可重複性
        # 模型參數
        "model_name": 'BERT',                            # 模型名稱
        "config": 'bert-base-uncased',                   # 使用的BERT模型配置
        "dropout": 0.1,                                  # Dropout率，用於防止過擬合)
        "activation": 'Prelu',                           # 激活函數類型
        "hidden_dim": 384,                               # 隱藏層維度
        # 訓練參數
        "learning_rate": 1e-4,                           # 學習率
        "epochs": 3,                                     # 訓練輪數
        "max_len": 512,                                  # 輸入文本的最大長度
        "batch_size": 16,                                # 每批次處理的數據量 (16、32、64)
        # 優化相關參數
        "early_stopping_patience": 7,                    # 早停耐心值：驗證損失多少輪未改善就停止
        "scheduler_patience": 3,                         # 學習率調整器的耐心值
        "scheduler_factor": 0.5,                         # 學習率調整的倍率
        "min_lr": 1e-6,                                  # 最小學習率
        "weight_decay": 0.01,                            # 權重衰減，用於正則化
        "warmup_steps": 0                                # 預熱步驟數
    }

# === 資料處理相關 ===
class CustomDataset(Dataset):
    def __init__(self, mode, df, specify, args):
        """初始化資料集"""
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.df = df.reset_index(drop=True)
        self.specify = specify
        if self.mode != 'test':
            self.label = self.df['label']
        
        self.tokenizer = AutoTokenizer.from_pretrained(args["config"])
        self.max_len = args["max_len"]
        self.num_class = args["num_class"]

    def __len__(self):
        """返回資料集大小"""
        return len(self.df) if self.df is not None else 0

    def one_hot_label(self, label):
        """將標籤轉換為 one-hot 編碼"""
        return F.one_hot(torch.tensor(label), num_classes=self.num_class)

    def tokenize(self, input_text):
        """將輸入文本轉換為模型所需的格式"""
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        )
        
        return (
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs["token_type_ids"]
        )

    def __getitem__(self, index):
        """獲取單一資料項"""
        try:
            if index >= len(self.df):
                raise IndexError(f"索引 {index} 超出範圍 (資料集大小: {len(self.df)})")
            
            sentence = str(self.df.iloc[index][self.specify])
            ids, mask, token_type_ids = self.tokenize(sentence)

            if self.mode == "test":
                return (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long)
                )
            else:
                if self.num_class > 2:
                    return (
                        torch.tensor(ids, dtype=torch.long),
                        torch.tensor(mask, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        self.one_hot_label(self.label.iloc[index])
                    )
                else:
                    return (
                        torch.tensor(ids, dtype=torch.long),
                        torch.tensor(mask, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        torch.tensor(self.label.iloc[index], dtype=torch.long)
                    )
        
        except Exception as e:
            logging.error(f"獲取數據時發生錯誤: index={index}, error={str(e)}")
            raise

def prepare_data(parameters, output_dir):
    """準備訓練、驗證和測試數據"""
    # 設定隨機種子
    torch.manual_seed(parameters['seed'])
    
    try:
        # 載入 IMDB 數據集
        print("正在載入 IMDB 數據集...")
        dataset = load_dataset("imdb")
        
        # 合併訓練和測試數據
        all_data = []
        
        # 處理訓練數據
        for data in dataset['train']:
            all_data.append({
                'text': data['text'],
                'label': data['label']
            })
            
        # 處理測試數據
        for data in dataset['test']:
            all_data.append({
                'text': data['text'],
                'label': data['label']
            })
        
        # 轉換為 DataFrame
        all_df = pd.DataFrame(all_data, columns=['text', 'label'])
        print(f"數據集總量: {len(all_df)} 筆")
        
        # 檢查標籤分布
        label_dist = all_df.label.value_counts() / len(all_df)
        print("\n標籤分布：")
        print(label_dist)
        
        # 分割數據集
        # 先分出訓練集
        train_df, temp_data = train_test_split(
            all_df,
            random_state=parameters['seed'],
            train_size=0.8
        )
        
        # 再從剩餘數據分出驗證集和測試集
        val_df, test_df = train_test_split(
            temp_data,
            random_state=parameters['seed'],
            train_size=0.5
        )
        
        # 採樣小部分數據用於快速測試（可選）
        SAMPLE_FRAC = 0.1  # 使用10%的數據
        train_df = train_df.sample(
            frac=SAMPLE_FRAC,
            random_state=parameters['seed']
        )
        val_df = val_df.sample(
            frac=SAMPLE_FRAC,
            random_state=parameters['seed']
        )
        
        print("\n數據集分割結果：")
        print(f"訓練集: {len(train_df)} 筆")
        print(f"驗證集: {len(val_df)} 筆")
        print(f"測試集: {len(test_df)} 筆")
        
        # 創建數據集實例
        train_dataset = CustomDataset('train', train_df, 'text', parameters)
        val_dataset = CustomDataset('val', val_df, 'text', parameters)
        test_dataset = CustomDataset('test', test_df, 'text', parameters)
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=parameters['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=parameters['batch_size'],
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=parameters['batch_size'],
            shuffle=False
        )
        
        # 儲存處理後的數據（可選）
        train_df.to_csv(os.path.join(output_dir, 'data', 'train.tsv'), sep='\t', index=False)
        val_df.to_csv(os.path.join(output_dir, 'data', 'val.tsv'), sep='\t', index=False)
        test_df.to_csv(os.path.join(output_dir, 'data', 'test.tsv'), sep='\t', index=False)
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"數據準備過程中發生錯誤：{str(e)}")
        raise e

class EarlyStopping:
    """早停機制實作"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience  # 容忍epoches數
        self.min_delta = min_delta  # 最小變化閾值
        self.counter = 0  # 計數器
        self.best_loss = None  # 最佳損失值
        self.early_stop = False  # 是否早停
        self.val_loss_min = np.inf  # 最小驗證損失
        
    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping 計數: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        '''儲存模型'''
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# === 模型定義 ===
class Dense(nn.Module):
    """全連接層的實作, 包含線性層、dropout和激活函數"""
    def __init__(self, input_dim, output_dim, dropout_rate, activation='tanh'):
        super(Dense, self).__init__()
        
        # 定義線性層
        self.hidden_layer = nn.Linear(input_dim, output_dim)
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        # 激活函數
        self.activation = self._get_activation(activation)
        # 初始化權重
        nn.init.xavier_uniform_(self.hidden_layer.weight)

    def _get_activation(self, activation):
        """根據指定的類型返回激活函數"""
        activation_dict = {
            'Prelu': nn.PReLU(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'LeakyReLU': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        return activation_dict.get(activation, nn.Tanh())

    def forward(self, inputs):
        """前向傳播"""
        # 線性變換
        logits = self.hidden_layer(inputs)
        # Dropout
        logits = self.dropout(logits)
        # 激活函數
        logits = self.activation(logits)
        
        return logits

class BertClassifier(BertPreTrainedModel):
   """BERT分類器模型"""
   def __init__(self, config, args):
       super(BertClassifier, self).__init__(config)
       # 載入預訓練BERT模型
       self.bert = BertModel(config)
       # 設定分類數量
       self.num_labels = args["num_class"]
       # Dropout層
       self.dropout = nn.Dropout(args["dropout"])
       # 分類器（線性層）
       self.classifier = nn.Linear(config.hidden_size, self.num_labels)
       # 初始化權重
       self.init_weights()

   def forward(self, 
               input_ids=None,
               attention_mask=None, 
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None):
       """前向傳播函數"""
       return_dict = return_dict if return_dict is not None else self.config.use_return_dict

       # BERT模型的輸出
       outputs = self.bert(
           input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids,
           position_ids=position_ids,
           head_mask=head_mask,
           inputs_embeds=inputs_embeds,
           output_attentions=output_attentions,
           output_hidden_states=output_hidden_states,
           return_dict=return_dict
       )
       
       # 獲取[CLS]標記的輸出
       # outputs[1]是pooler_output，對應[CLS]標記的表示
       pooled_output = outputs[1]
       
       # 添加dropout層
       pooled_output = self.dropout(pooled_output)
       
       # 線性分類層
       logits = self.classifier(pooled_output)

       return logits

   def predict(self, text, tokenizer, device, max_len=512):
       """預測單句文本的方法"""
       # 設定為評估模式
       self.eval()
       
       # 文本轉換為模型輸入格式
       encoding = tokenizer.encode_plus(
           text,
           max_length=max_len,
           padding='max_length',
           truncation=True,
           return_tensors='pt'
       )
       
       # 將輸入移至指定設備
       inputs = {
           'input_ids': encoding['input_ids'].to(device),
           'attention_mask': encoding['attention_mask'].to(device),
           'token_type_ids': encoding['token_type_ids'].to(device)
       }
       
       # 禁用梯度計算
       with torch.no_grad():
           outputs = self(**inputs)
           
       return outputs

# === 訓練與評估 ===
def train(model, train_loader, val_loader, optimizer, args, device, output_dir, patience=7):
   """模型訓練函數"""
   model = model.to(DEVICE)
   # 初始化記錄字典
   metrics = ['loss', 'acc', 'f1', 'rec', 'prec']
   mode = ['train_', 'val_']
   record = {s + m: [] for s in mode for m in metrics}
   
   # 定義損失函數
   loss_fct = nn.CrossEntropyLoss()

   # 初始化早停機制
   early_stopping = EarlyStopping(patience=patience, min_delta=0)
   best_model_path = os.path.join(output_dir, 'models', f"best_{args['model_name']}_{args['time'].split('_')[0]}.pt")
   
   # 初始化學習率調整器
   scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',          
        factor=0.5,          
        patience=3,          
        verbose=True,        
        min_lr=1e-6         
    )
   
   print("開始訓練...")
   for epoch in range(args["epochs"]):
       st_time = time.time()
       
       # 初始化訓練指標
       train_loss, train_acc = 0.0, 0.0
       train_f1, train_rec, train_prec = 0.0, 0.0, 0.0
       step_count = 0
       
       # 訓練模式
       model.train()
       for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}"):
           # 將數據移至指定設備
           ids, masks, token_type_ids, labels = [t.to(device) for t in data]
           
           # 清空梯度
           optimizer.zero_grad()
           
           # 前向傳播
           logits = model(
               input_ids=ids,
               token_type_ids=token_type_ids,
               attention_mask=masks
           )
           
           # 計算損失和指標
           loss = loss_fct(logits, labels)
           acc, f1, rec, prec = cal_metrics(get_pred(logits), labels, 'macro')
           
           # 反向傳播
           loss.backward()
           optimizer.step()
           
           # 累計指標
           train_loss += loss.item()
           train_acc += acc
           train_f1 += f1
           train_rec += rec
           train_prec += prec
           step_count += 1
           
           # 打印訓練進度
           if step_count % 100 == 0:
               print(f"Epoch {epoch+1}, Step {step_count}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
       
       # 計算平均值
       train_loss = train_loss / step_count
       train_acc = train_acc / step_count
       train_f1 = train_f1 / step_count
       train_rec = train_rec / step_count
       train_prec = train_prec / step_count
       
       # 驗證模型並獲取驗證指標
       val_metrics = evaluate(model, val_loader, device)
       val_loss, val_acc, val_f1, val_rec, val_prec = val_metrics

       # 更新學習率
       scheduler.step(val_loss)
       current_lr = optimizer.param_groups[0]['lr']
       print(f'當前學習率：{current_lr}')

       # 檢查是否需要早停
       early_stopping(val_loss, model, best_model_path)
       if early_stopping.early_stop:
           print("觸發早停機制！")
           break
       
       # 記錄訓練指標
       record['train_loss'].append(train_loss)
       record['train_acc'].append(train_acc)
       record['train_f1'].append(train_f1)
       record['train_rec'].append(train_rec)
       record['train_prec'].append(train_prec)
       
       # 記錄驗證指標
       record['val_loss'].append(val_loss)
       record['val_acc'].append(val_acc)
       record['val_f1'].append(val_f1)
       record['val_rec'].append(val_rec)
       record['val_prec'].append(val_prec)
       
       # 打印當前epoch的結果
       print(f'[epoch {epoch + 1}] 耗時: {time.time() - st_time:.4f} 秒')
       print('         loss     acc     f1      rec     prec')
       print('train | %.4f, %.4f, %.4f, %.4f, %.4f' % 
             (train_loss, train_acc, train_f1, train_rec, train_prec))
       print('val   | %.4f, %.4f, %.4f, %.4f, %.4f\n' % 
             (val_loss, val_acc, val_f1, val_rec, val_prec))
       
       # 儲存最終模型
       if epoch == args["epochs"] - 1:
           save_path = os.path.join(
               output_dir, 
               'models', 
               f"{args['model_name']}_{args['time'].split('_')[0]}.pt"
           )
           torch.save(model.state_dict(), save_path)
           print(f"模型已儲存至 {save_path}")
   
   # 載入最佳模型
   model.load_state_dict(torch.load(best_model_path))
   return record

def evaluate(model, data_loader, device):
    """評估模型性能的函數"""
    model = model.to(DEVICE)
    # 初始化指標
    all_predictions = []
    all_labels = []
    val_loss = 0.0
    val_acc, val_f1, val_rec, val_prec = 0.0, 0.0, 0.0, 0.0
    step_count = 0
    
    # 定義損失函數
    loss_fct = nn.CrossEntropyLoss()
    
    # 設定為評估模式
    model.eval()
    
    # 禁用梯度計算
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            # 將資料移至指定設備
            ids, masks, token_type_ids, labels = [t.to(device) for t in data]
            
            # 取得模型預測結果
            logits = model(
                input_ids=ids,
                token_type_ids=token_type_ids,
                attention_mask=masks
            )
            
            # 計算損失和指標
            loss = loss_fct(logits, labels)
            acc, f1, rec, prec = cal_metrics(get_pred(logits), labels, 'macro')
            
            # 儲存預測結果和標籤
            predictions = get_pred(logits)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 累計指標
            val_loss += loss.item()
            val_acc += acc
            val_f1 += f1
            val_rec += rec
            val_prec += prec
            step_count += 1
    
    # 計算平均值
    val_loss = val_loss / step_count
    val_acc = val_acc / step_count
    val_f1 = val_f1 / step_count
    val_rec = val_rec / step_count
    val_prec = val_prec / step_count
    
    return val_loss, val_acc, val_f1, val_rec, val_prec

# === 評估指標計算 ===
def cal_metrics(pred, ans, method='macro'):
    """計算各項評估指標"""
    try:
        # 檢查輸入類型並轉換為 numpy array
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(ans, torch.Tensor):
            ans = ans.detach().cpu().numpy()
        
        # 確保資料類型為 int64
        pred = pred.astype(np.int64)
        ans = ans.astype(np.int64)
        
        # 計算各項指標
        accuracy = accuracy_score(ans, pred)
        f1 = f1_score(ans, pred, average=method, zero_division=0)
        recall = recall_score(ans, pred, average=method, zero_division=0)
        precision = precision_score(ans, pred, average=method, zero_division=0)
        
        return accuracy, f1, recall, precision
        
    except Exception as e:
        logging.error(f"計算評估指標時發生錯誤: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def get_pred(logits):
   """將模型輸出轉換為預測類別"""
   return torch.argmax(logits, dim=1)

# === 視覺化工具 ===
def draw_pic(record, name, output_dir, img_save=False, show=False):
    """繪製學習曲線圖表"""
    # 使用 matplotlib 內建風格
    plt.style.use('default')
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(record['train_' + name]) + 1)
    
    # 繪製訓練曲線（使用藍色）
    plt.plot(epochs, record['train_' + name], 
             color='#1f77b4',  # 標準藍色
             label=f'Training {name}', 
             linewidth=2, 
             marker='o',
             markersize=6)
    
    # 繪製驗證曲線（使用橙色）
    plt.plot(epochs, record['val_' + name], 
             color='#ff7f0e',  # 標準橙色
             label=f'Validation {name}', 
             linewidth=2, 
             marker='o',
             markersize=6)
    
    # 設定標題和標籤
    plt.title(f'{name.capitalize()} Curves', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(name.capitalize(), fontsize=14)
    
    # 添加網格（淡灰色）
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 設定圖例
    plt.legend(fontsize=12, loc='best')
    
    # 調整布局
    plt.tight_layout()
    
    # 儲存圖片
    if img_save:
        save_path = os.path.join(output_dir, 'plots', f'{name}_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"圖片已儲存至: {save_path}")
    
    # 顯示圖片
    if show:
        plt.show()
    
    # 關閉圖表以釋放記憶體
    plt.close()

# === 設定輸出路徑 ===
def setup_output_dir(parameters):
    """設定輸出資料夾"""
    # 建立以時間命名的輸出資料夾
    output_dir = os.path.join(
        'outputs', 
        f"{parameters['model_name']}_{parameters['time'].replace(':', '-')}"
    )
    
    # 建立資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立子資料夾
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)  # 存放處理後的數據
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)  # 存放模型
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)  # 存放圖表
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)  # 存放日誌
    
    return output_dir

# === 預測功能 ===
def predict_one(query, model, parameters):
   """預測單句文本的情緒"""
   try:
       # 初始化分詞器
       tokenizer = AutoTokenizer.from_pretrained(parameters['config'])
       
       # 設定模型為評估模式
       model = model.to(DEVICE)
       model.eval()
       
       # 在不計算梯度的情況下進行預測
       with torch.no_grad():
           # 將文本轉換為模型輸入格式
           inputs = tokenizer.encode_plus(
               query,
               max_length=parameters['max_len'],
               truncation=True,
               padding='max_length',
               return_tensors='pt'
           )
           
           # 將輸入移至指定設備
           input_ids = inputs['input_ids'].to(DEVICE)
           attention_mask = inputs['attention_mask'].to(DEVICE)
           token_type_ids = inputs['token_type_ids'].to(DEVICE)
           
           # 模型前向傳播
           logits = model(
               input_ids=input_ids,
               token_type_ids=token_type_ids,
               attention_mask=attention_mask
           )
           
           # 計算機率分布
           probs = torch.softmax(logits, dim=1)
           
           # 獲取預測類別
           pred = torch.argmax(probs[0], dim=0).item()
           
           return probs, pred
           
   except Exception as e:
       print(f"預測過程中發生錯誤: {str(e)}")
       return None, None

def predict(data_loader, model, parameters):
   """批量預測函數"""
   model = model.to(DEVICE)
   total_probs = []
   total_pred = []
   
   try:
       model.eval()
       with torch.no_grad():
           for data in tqdm(data_loader, desc="Predicting"):
               input_ids, attention_mask, token_type_ids = [t.to(DEVICE) for t in data]
               
               logits = model(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids
               )
               
               probs = torch.softmax(logits, dim=1)
               pred = torch.argmax(probs, dim=1)
               
               total_probs.extend(probs.cpu().numpy())
               total_pred.extend(pred.cpu().numpy())
       
       return np.array(total_probs), np.array(total_pred)
   
   except Exception as e:
       logging.error(f"批量預測時發生錯誤: {str(e)}")
       return None, None

def save_predictions(predictions, original_df, output_path='predictions.tsv'):
   """儲存預測結果"""
   try:
       # 複製原始數據
       result_df = original_df.copy()
       
       # 添加預測結果
       result_df['predicted_label'] = predictions
       
       # 儲存結果
       result_df.to_csv(output_path, sep='\t', index=False)
       print(f"預測結果已儲存至: {output_path}")
       
   except Exception as e:
       print(f"儲存預測結果時發生錯誤: {str(e)}")

# === 工具函數 ===
def Softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

def label2class(label):
    return {0: 'negative', 1: 'positive'}[label]

# === 主程式 ===
def main():
    try:
        start_time = time.time()
        # 設定參數
        parameters = get_parameters()

        # 設定隨機種子
        set_seed(parameters['seed'])

        # 設定輸出路徑
        output_dir = setup_output_dir(parameters)

        # 設定日誌
        setup_logging(output_dir)
        logger = logging.getLogger(__name__)
        
        # 記錄參數
        with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
            json.dump(parameters, f, indent=4)
        
        warnings.filterwarnings('ignore')
        logger.info(f"使用設備: {DEVICE}")
        logger.info(f"輸出路徑: {output_dir}")

        # 準備資料
        logger.info("開始準備資料...")
        train_loader, val_loader, test_loader = prepare_data(parameters, output_dir)

        # 初始化模型
        logger.info("初始化模型...")
        model = BertClassifier.from_pretrained(parameters['config'], parameters).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])

        # 訓練模型
        logger.info("開始訓練...")
        history = train(model, train_loader, val_loader, optimizer, parameters, DEVICE, output_dir, patience=7)
        
        # 繪製學習曲線
        logger.info("繪製學習曲線...")
        metrics = ['loss', 'acc', 'f1', 'rec', 'prec']
        for metric in metrics:
            draw_pic(history, metric, output_dir, img_save=True, show=True)
        
        # 進行預測測試
        test_text = "This movie doesn't attract me"
        probs, pred = predict_one(test_text, model, parameters)
        logger.info(f"測試文本: {test_text}")
        logger.info(f"預測結果: {label2class(pred)}")
        logger.info(f"機率分布: {probs}")
        
        # 總訓練時間
        total_time = time.time() - start_time
        logger.info(f"總訓練時間：{total_time/60:.2f} 分鐘")

    except Exception as e:
        logger.error(f"程式執行發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()