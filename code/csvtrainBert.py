import pandas as pd
import torch, os
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel, BertForSequenceClassification

# 設置讀取檔案的路徑和檔名
# file_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/ChineseNlpCorpus/weibo_senti_100k/weibo_senti_100k/chinese.csv"
# save_path = "C:/Users/ROUSER6/Desktop\DEEP_LEARNING_Pratice/model/useCsvttrainModel"

# Loding token
#tokenizer = BertTokenizer.from_pretrained(save_path)
# Loding Model
#model = LongformerModel.from_pretrained(save_path)

def  trainModel(file_path, tokenizer, model, save_path):
    # 讀取 csv 檔案並將資料轉換為 DataFrame
    data = pd.read_csv(file_path, encoding='utf-8')

    # 將文本轉換為 BERT token ID 序列，並添加特殊標記
    input_ids = []
    attention_masks = []
    for text in data['text']:
        encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # 創建 PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(data['label'], dtype=torch.long)

    # 將資料集包裝為 TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    return dataset

# dataset=trainModel(file_path, tokenizer)
# print(len(dataset))