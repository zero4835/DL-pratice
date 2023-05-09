import os, torch, glob
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel

# 设置数据集目录路径
data_dir = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/THUCNews/THUCNews/test"
#transferTochinese
save_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/model"

# 加载 tokenizer，并载入 BERT 预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# 定义数据处理函数
def load_dataset(data_dir, tokenizer):
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
    return dataset

# 加载数据集
dataset = load_dataset(data_dir, tokenizer)

print("len(dataset) = ", len(dataset))

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#定义数据加载器
batch_size = 32
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
test_sampler = SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)