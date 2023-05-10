import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_sentiment(input_text_list, tokenizer, model):
    # 将多个句子转换成PyTorch tensors
    encoded_dict = tokenizer.batch_encode_plus(input_text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    
    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs[0]
        predicted_labels = torch.argmax(logits, axis=1)
    
    # 将预测结果转换成情感标签
    predicted_scores = predicted_labels.tolist()
    return predicted_scores

# 加载预训练的BERT模型和分词器
from transformers import BertTokenizer,LongformerTokenizer, LongformerModel

data_dir = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/THUCNews/THUCNews/transferTochinese"
save_path = "C:/Users/ROUSER6/Desktop/DEEP_LEARNING_Pratice/model/useTxTtrainModel"

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#stokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
tokenizer = BertTokenizer.from_pretrained(save_path)

model = LongformerModel.from_pretrained(save_path)

# 输入文本列表
input_text_list = ["这部電影真是太棒了！", "我真的很喜歡這個產品。", "這個餐廳的食物真的很不好吃"]

# 预测情感标签
predicted_scores = predict_sentiment(input_text_list, tokenizer, model)

# 打印预测结果
for i, text in enumerate(input_text_list):
    if predicted_scores[i] == 0:
        print("文本 '{}' 的情感分析结果为：負面情感".format(text))
    else:
        print("文本 '{}' 的情感分析结果为：正面情感".format(text))