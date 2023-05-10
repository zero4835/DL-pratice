import csvtrainBert as CTB
import os, torch, glob
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel, BertForSequenceClassification



# 設置讀取檔案的路徑和檔名
file_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/ChineseNlpCorpus/weibo_senti_100k/weibo_senti_100k/chinese.csv"
save_path = "C:/Users/ROUSER6/Desktop\DEEP_LEARNING_Pratice/model/useCsvttrainModel"

# Loding token
tokenizer = BertTokenizer.from_pretrained(save_path)
# Loding Model
model =BertForSequenceClassification.from_pretrained(save_path)

# dataset = CTB.trainModel(file_path, tokenizer, model, save_path) 
# print(len(dataset))

# 定義類別標籤
label_list = ['Negative', 'Positive']

# 輸入文本
text = "和女兒久違的輕旅行享受夏天的開幕 陽光實在很熱情中南部走走繞繞 吃吃喝喝 享受愜意時光  再計畫著下一次的旅途☺️"

# 將文本轉換為 BERT token ID 序列，並添加特殊標記
encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_attention_mask=True)

# 創建 PyTorch tensors
input_ids = torch.tensor([encoded_dict['input_ids']], dtype=torch.long)
attention_masks = torch.tensor([encoded_dict['attention_mask']], dtype=torch.long)

# 使用模型進行預測
outputs = model(input_ids, attention_mask=attention_masks)
_, predicted = torch.max(outputs.logits, dim=1)

# 輸出預測結果
print(label_list[predicted])    