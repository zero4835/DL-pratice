import os, torch, glob
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel

save_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/model"

# 定义数据处理函数
def load_dataset(data_dir, tokenizer):
    #Loding Model
    model = LongformerModel.from_pretrained(save_path)
    
    texts = []
    labels = []
    label_dict = {}
    label_id = 0
    
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            label = os.path.basename(os.path.dirname(filepath))
            texts.append(text)
            if label not in label_dict:
                label_dict[label] = label_id
                label_id += 1
            labels.append(label_dict[label])
            
    # 将文本转换为 BERT token ID 序列，并添加特殊标记
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=4096,
                                            padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])


    # 创建 PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
        
    # 将数据集包装成 TensorDataset
    dataset = TensorDataset(input_ids, labels)
    
    #將訓練完的Model與Tokenizer保存
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Complete!")
    
    return dataset

# Test load
# dataset = load_dataset(data_dir, tokenizer)