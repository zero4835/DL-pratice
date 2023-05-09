import torch
import trainBert as TB
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel

data_dir = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/THUCNews/THUCNews/test"
save_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/model"

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
tokenizer = BertTokenizer.from_pretrained(save_path)

model = LongformerModel.from_pretrained(save_path)

#dataset = TB.load_dataset(data_dir, tokenizer)

def predict_sentiment(input_text_list, tokenizer, model):
    # 將多個句子轉換成 PyTorch tensors
    encoded_dict = tokenizer.batch_encode_plus(input_text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    
    # 使用模型進行預測
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs[0]
        predicted_labels = torch.argmax(logits, axis=1)
    
    # 將預測結果轉換成情緒分數
    predicted_scores = predicted_labels.tolist()

    return predicted_scores


input_text_list = ["這部電影真是太棒了！", "我真的很喜歡這個產品。", "這餐廳的食物真的很不好吃。"]
print(predict_sentiment(input_text_list, tokenizer, model))
