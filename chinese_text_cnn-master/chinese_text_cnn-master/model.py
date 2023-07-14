import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        # num_embeddings (python:int) – 辭典的大小尺寸，比如一共出現5000個詞，則輸入5000。此時索引為（0-4999）
        # embedding_dim (python:int) – 嵌入向量的維度，即用多少維来表示一个符號。
        # 输入: (∗) , 包含提取數量的任意形狀的長整型張量。
        # 输出: (∗,H) , 其中 * 為輸入的形狀，H為embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        # in_channel:　輸入數據的通道數，例如RGB圖片通道數為3；
        # out_channel: 輸出數據的通道數，此根據模型調整；
        # kennel_size: 卷積和大小
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        # x shape=[128, 55]=[batch_size, sentence_len]
        if self.embedding2:

            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            # x shape=[128, 55, 128] = [batch_size, sentence_len, embedding_dimension]
            x = self.embedding(x)
            # x shape=[128, 1, 55, 128] = [batch_size, 1, sentence_len, embedding_dimension]
            x = x.unsqueeze(1)

        # x[0] shape = [128, 100, 52] = [batch_size, filter_num, sentence_len-filter_sizes]
        # x[1] shape = [128, 100, 51] = [batch_size, filter_num, sentence_len-filter_sizes]
        # x[2] shape = [128, 100, 50] = [batch_size, filter_num, sentence_len-filter_sizes]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # x[0] shape=[128, 100]
        # x[1] shape=[128, 100]
        # x[2] shape=[128, 100]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]

        # x shape=[128, 300]
        x = torch.cat(x, 1)

        x = self.dropout(x)

        # logits shape=[128, 3]=[batch_size, class_num]
        logits = self.fc(x)
        return logits
