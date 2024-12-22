# 導入必要的套件
import os
import torch
import logging
import warnings
from transformers import AutoTokenizer
from train_Bert import BertClassifier, get_parameters

# 設定警告級別
warnings.filterwarnings('ignore')

# 設定全局設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, parameters):
    """載入訓練好的模型"""
    try:
        # 初始化模型
        model = BertClassifier.from_pretrained(parameters['config'], parameters)
        
        # 載入模型權重
        model.load_state_dict(torch.load(model_path))
        model = model.to(DEVICE)
        model.eval()
        
        return model
    
    except Exception as e:
        logging.error(f"載入模型時發生錯誤：{str(e)}")
        raise

def predict_sentiment(text, model, tokenizer, parameters):
    """預測文本情緒"""
    try:
        # 將文本轉換為模型輸入格式
        inputs = tokenizer.encode_plus(
            text,
            max_length=parameters['max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 將輸入移至指定設備
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        token_type_ids = inputs['token_type_ids'].to(DEVICE)
        
        # 進行預測
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        # 計算機率值
        probabilities = torch.softmax(outputs, dim=1)
        
        # 獲取預測類別
        predicted_class = torch.argmax(probabilities).item()
        
        return predicted_class, probabilities[0].tolist()
    
    except Exception as e:
        logging.error(f"預測過程中發生錯誤：{str(e)}")
        raise

def setup_logging():
    """設定日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    try:
        # 設定日誌
        setup_logging()
        
        # 設定參數
        parameters = get_parameters()
        
        # 載入已訓練的模型
        model_path = "best_BERT.pt"
        model = load_model(model_path, parameters)
        
        # 初始化分詞器
        tokenizer = AutoTokenizer.from_pretrained(parameters['config'])
        
        print("模型已載入完成，可以開始預測了！")
        print("請輸入文本進行情緒分析（輸入 'q' 退出）")
        
        while True:
            # 使用者輸入文本
            text = input("\n請輸入要分析的文本：")
            
            if text.lower() == 'q':
                print("感謝使用！再見！")
                break
            
            # 進行情緒分析
            predicted_class, probabilities = predict_sentiment(
                text, model, tokenizer, parameters
            )
            
            # 輸出結果
            sentiment_labels = ['負面', '正面']
            print(f"\n預測結果：{sentiment_labels[predicted_class]}")
            print(f"信心度：{max(probabilities)*100:.2f}%")
            print(f"完整機率分布：")
            for label, prob in zip(sentiment_labels, probabilities):
                print(f"{label}：{prob*100:.2f}%")
    
    except Exception as e:
        logging.error(f"程式執行發生錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main()