import torch, os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer

# 設置數據集目錄路徑
data_dir = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/IMDB/aclImdb/train"

# 加載 tokenizer，並載入 BERT 預訓練模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加載數據集
def load_dataset(data_dir, tokenizer):
    pos_dir = os.path.join(data_dir, "pos")
    neg_dir = os.path.join(data_dir, "neg")

    pos_data = []
    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), "r", encoding="utf-8") as f:
            pos_data.append(f.read())

    neg_data = []
    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), "r", encoding="utf-8") as f:
            neg_data.append(f.read())

    texts = pos_data + neg_data
    labels = [1]*len(pos_data) + [0]*len(neg_data)

    # 將文本轉換為 BERT token ID 序列，並添加特殊標記
    input_ids = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=True, max_length=1024, truncation=True)
        input_ids.append(encoded)

    # 創建 PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # 將數據集包裝成 TensorDataset
    dataset = TensorDataset(input_ids, labels)
    return dataset

dataset = load_dataset(data_dir, tokenizer)
