import argparse
import torch
import torchtext.data as data
from torchtext.vocab import Vectors

import model
import train
import dataset


parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='model', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    # train_dataset TabularDataset:56700
    # dev_dataset TabularDataset:7000
    train_dataset, dev_dataset = dataset.get_dataset('data', text_field, label_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    vocab = text_field.vocab
    # text_field vocab={Vocab:35572}
    # label_field vocab={Vocab:3}
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    # train_iter={Iterator:443} batch_size:128
    # dev_iter={Iterator:1} batch_size:7001
    return train_iter, dev_iter, vocab


def predict(model, vocab, sentence):
    model.eval()
    sentence = [vocab.stoi[w] for w in dataset.word_cut(sentence)]
    if len(sentence) < 5:
        for i in range(5):
            sentence.append(1)
    sentence = torch.as_tensor(sentence)
    sentence = sentence.unsqueeze(1).t()
    logits = model(sentence)
    #return [pos, neu, neg]
    print(logits)
    max_value, max_index = torch.max(logits, dim=1)

    # 取得最大值的分數和索引
    max_score = max_value.item()
    max_emotion_index = max_index.item()

    # 根據索引對應到情緒類別
    emotion_classes = ['正面情緒', '中性情緒', '負面情緒']
    max_emotion = emotion_classes[max_emotion_index]

    print('最大分數:', max_score)
    print('最大情緒:', max_emotion)
    print(end='\n')

print('Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
# train_iter={Iterator:443} batch_size:128
# dev_iter={Iterator:1} batch_size:7000
train_iter, dev_iter, vocab = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
# 35572
args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
# 3
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))
    
model_path = './snapshot/best_steps_1100.pt'
# text_cnn = model.TextCNN(model_path)
text_cnn = model.TextCNN(args)

if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    text_cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()
    
# -----------------------------------------------------------------
# state_dict = torch.load(model_path)
# # state_dict = torch.load(args.)

# # 將加載的狀態字典分配給模型實例的 state_dict 屬性
# text_cnn.load_state_dict(state_dict)

# # 評估模式
# text_cnn.eval()
# print("很開心")
# predict(text_cnn, vocab, '開心')
# print("不開心")
# predict(text_cnn, vocab, '不開心')
# print("今天的天氣不錯")
# predict(text_cnn, vocab, '今天的天氣不錯')
# print("操控性好，性價比高，油耗低")
# predict(text_cnn, vocab, '操控性好，性價比高，油耗低')

# training
try:
    train.train(train_iter, dev_iter, text_cnn, args) 
except KeyboardInterrupt:
    print('Exiting from training early')
