# BERT 情感分析模型

這是一個基於 BERT 的情感分析模型，可以分析文本的情感傾向（正面/負面）。

## 模型下載

由於模型檔案較大，我們將其存放在 Hugging Face Model Hub，您可以從以下連結下載：

[下載模型](https://huggingface.co/MatchaCat4477/best_BERT/resolve/main/best_BERT.pt)

下載後，請將模型檔案 `best_BERT.pt` 放在專案根目錄下。

## 環境設置

1. 安裝必要套件：
```bash
pip install torch transformers datasets pandas numpy tqdm scikit-learn matplotlib
```

2. 下載模型檔案並放置於正確位置。

## 使用方法

### 單一文本預測

使用 `predict.py` 進行單一文本的情感分析：

```bash
python predict.py
```

程式會進入互動模式，您可以輸入想要分析的文本。輸入 'q' 可以退出程式。

### 批量測試

使用 `test_sentiment.py` 進行批量文本測試：

```bash
python test_sentiment.py
```

## 檔案說明

- `train_Bert.py`: 模型訓練主程式
- `predict.py`: 單一文本預測程式
- `test_sentiment.py`: 批量測試程式
- `best_BERT.pt`: 訓練好的模型檔案（需要從 Hugging Face 下載）

## 預測結果範例

模型輸出會包含以下資訊：
- 預測的情感傾向（正面/負面）
- 預測的信心度
- 完整的機率分布

例如：
```
預測結果：負面
信心度：96.90%
完整機率分布：
負面：96.90%
正面：3.10%
```

## 注意事項

1. 確保您有足夠的硬碟空間（模型檔案約 418MB）
2. 建議使用 GPU 進行預測，可以大幅提升處理速度
3. 第一次執行時會下載 BERT 基礎模型，需要網路連接

## 錯誤排除

如果遇到 "RuntimeError: Expected all tensors to be on the same device" 錯誤，請確保：
1. 檢查 CUDA 是否可用
2. 確認所有張量都在同一個設備上（CPU 或 GPU）

如果有任何問題，歡迎提出 Issue。
