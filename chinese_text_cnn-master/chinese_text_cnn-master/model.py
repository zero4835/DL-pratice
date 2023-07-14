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

        # num_embeddings (python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
        # embedding_dim (python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
        # 输入: (∗) , 包含提取的编号的任意形状的长整型张量。
        # 输出: (∗,H) , 其中 * 为输入的形状，H为embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        # in_channel:　输入数据的通道数，例RGB图片通道数为3；
        # out_channel: 输出数据的通道数，这个根据模型调整；
        # kennel_size: 卷积核大小
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
