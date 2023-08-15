import argparse
import pandas as pd
import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import torch.nn.functional as F

import model
import train
import dataset

parser = argparse.ArgumentParser(description="TextCNN text classifier")
# learning
parser.add_argument(
    "-lr", type=float, default=0.001, help="initial learning rate [default: 0.001]"
)
parser.add_argument(
    "-epochs", type=int, default=256, help="number of epochs for train [default: 256]"
)
parser.add_argument(
    "-batch-size", type=int, default=128, help="batch size for training [default: 128]"
)
parser.add_argument(
    "-log-interval",
    type=int,
    default=1,
    help="how many steps to wait before logging training status [default: 1]",
)
parser.add_argument(
    "-test-interval",
    type=int,
    default=100,
    help="how many steps to wait before testing [default: 100]",
)
parser.add_argument(
    "-save-dir", type=str, default="snapshot", help="where to save the snapshot"
)
parser.add_argument(
    "-early-stopping",
    type=int,
    default=1500,
    help="iteration numbers to stop without performance increasing",
)
parser.add_argument(
    "-save-best",
    type=bool,
    default=True,
    help="whether to save when get best performance",
)
# model
parser.add_argument(
    "-dropout",
    type=float,
    default=0.5,
    help="the probability for dropout [default: 0.5]",
)
parser.add_argument(
    "-max-norm",
    type=float,
    default=3.0,
    help="l2 constraint of parameters [default: 3.0]",
)
parser.add_argument(
    "-embedding-dim",
    type=int,
    default=256,
    help="number of embedding dimension [default: 128]",
)
parser.add_argument(
    "-filter-num",
    type=int,
    default=200,
    help="number of each size of filter[default: 100]",
)
parser.add_argument(
    "-filter-sizes",
    type=str,
    default="3,4,5",
    help="comma-separated filter sizes to use for convolution",
)

parser.add_argument(
    "-static",
    type=bool,
    default=False,
    help="whether to use static pre-trained word vectors",
)
parser.add_argument(
    "-non-static",
    type=bool,
    default=False,
    help="whether to fine-tune static pre-trained word vectors",
)
parser.add_argument(
    "-multichannel",
    type=bool,
    default=False,
    help="whether to use 2 channel of word vectors",
)
parser.add_argument(
    "-pretrained-name",
    type=str,
    default="sgns.zhihu.word",
    help="filename of pre-trained word vectSors",
)
parser.add_argument(
    "-pretrained-path",
    type=str,
    default="pretrained",
    help="path of pre-trained word vectors",
)

# device
parser.add_argument(
    "-device",
    type=int,
    default=-1,
    help="device to use for iterate data, -1 mean cpu [default: -1]",
)

# option
parser.add_argument(
    "-snapshot",
    type=str,
    default='./snapshot/best_steps_200.pt',
    help="filename of model snapshot [default: None]",
)
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset, test_dataset = dataset.get_dataset("data", text_field, label_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset), len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs,
    )
    return train_iter, dev_iter, test_iter


def predict(model, vocab, sentence):
    model.eval()
    sentence = [vocab.stoi[w] for w in dataset.word_cut(sentence)]
    if len(sentence) < 5:
        for i in range(5):
            sentence.append(1)
    sentence = torch.as_tensor(sentence)
    sentence = sentence.unsqueeze(1).t()
    logits = model(sentence)
    print(logits)
    probs = F.softmax(logits, dim=1)
    # return [pos, neg]
    # print(probs)
    max_value, max_index = torch.max(probs, dim=1)

    # 取得最大值的分數和索引
    print(f"max_value.item(): {max_value.item()}")
    max_score = max_value.item()
    max_emotion_index = max_index.item()
    print(f"max emotion index: {max_emotion_index}")
    # 根據索引對應到情緒類別
    emotion_classes = ["正面", "負面", "中立"]
    emotion = emotion_classes[max_emotion_index]

    return max_score, emotion

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    true_emotion_classes = ["正面", "中立", "負面"]
    emotion_classes = ["中立", "負面", "正面", "<Unk>"]
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        
        # Print test text and predicted emotion for each sample in the batch
        for text, true_label, pred_label in zip(feature, target, torch.argmax(logits, dim=1)):
            words = [text_field.vocab.itos[word] for word in text.tolist() if word != text_field.vocab.stoi['<pad>']]
            text = " ".join(words)
            true_emotion = true_emotion_classes[true_label.item()]
            pred_emotion = emotion_classes[pred_label.item()]
            print(f"Test Text: {text}")
            print(f"True label item: {true_label.item()}")
            print(f"True Emotion: {true_emotion}")
            print(f"Predicted Emotion: {pred_emotion}\n")
            
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return float('{:.4f}'.format(accuracy.item()))

def test(model, vocab):
    data = pd.read_csv("./data/self_test.tsv", sep="\t")
    # data = pd.read_csv("./data/test_250.tsv", sep="\t")
    model.eval()
    emotion_classes = ["正面", "中立", "負面"]
    for index, row in data.iterrows():
        label = row["label"]  
        text = row["text"] 
        score, emotion = predict(model, vocab, text)
        print(f"文本: {text}")
        print(f"最大分數: {score}")
        print(f"Label: {emotion_classes[label]}")
        print(f"最大情緒: {emotion}\n")
    
    # eval(test_iter, model, args)

print("Loading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_dataset(
    text_field, label_field, args, device=-1, repeat=False, shuffle=True
)

args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
    
# classification 3
args.class_num = len(label_field.vocab) - 1

print(f"label_field.vocab: {len(label_field.vocab)}")

print("label_field.vocab\n\n")
print(label_field.vocab.itos)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(",")]

print("Parameters:")
for attr, value in sorted(args.__dict__.items()):
    if attr in {"vectors"}:
        continue
    print("\t{}={}".format(attr.upper(), value))

text_cnn = model.TextCNN(args)

if args.snapshot:
    print("\nLoading model from {}...\n".format(args.snapshot))

    pretrained_dict = torch.load(args.snapshot)

    # pretrained_vocabulary_size = pretrained_dict['embedding.weight'].shape[0]
    # args.vocabulary_size = pretrained_vocabulary_size

    # pretrained_embedding_dim = pretrained_dict['embedding.weight'].shape[1]
    # args.embedding_dim = pretrained_embedding_dim

    text_cnn = model.TextCNN(args)
    model_dict = text_cnn.state_dict()

    # 不匹配層去除
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'embedding' not in k)}
    # 更新參數權重
    # model_dict.update(pretrained_dict)
    # text_cnn.load_state_dict(model_dict)
    text_cnn.load_state_dict(pretrained_dict)

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()

# model_path = './snapshot/best_steps_100.pt'
# state_dict = torch.load(model_path)
# text_cnn.load_state_dict(state_dict)
test(text_cnn, text_field.vocab)


# try:
#     train.train(train_iter, dev_iter, text_cnn, args)
# except KeyboardInterrupt:
#     print('Exiting from training early')
