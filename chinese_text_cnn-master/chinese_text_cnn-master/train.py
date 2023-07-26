import os
import sys
import torch
import torch.nn.functional as F
import torch.autograd as autograd

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    
    best_acc_file = os.path.join('./model', 'best_accuracy.txt')
    if os.path.exists(best_acc_file):
        with open(best_acc_file, 'r') as f:
            best_acc = float(f.read())
    
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            # batch shape=[51, 128]
            # feature shape=[51, 128]
            # target shape=[128]
            feature, target = batch.text, batch.label
            # feature.data.t_(), target.data.sub_(1)
            with torch.no_grad():
                feature.t_(), target.sub_(1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, './model', 'best', steps, best_acc)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # feature.data.t_(), target.data.sub_(1)
        with torch.no_grad():
            feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        # feature shape=[7000, 59]
        # target shape=[7000]
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def predict(text, model, text_field, label_feild, args, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    text = text_field.preprocess(text)
    max_filter_size = max(args.filter_sizes)
    if len(text) < max_filter_size:
        pad_size = max_filter_size - len(text)
        text = text + [text_field.pad_token] * pad_size
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    output = model(x)
    _, predicted = torch.max(output, 1)
    print(output)
    return label_feild.vocab.itos[predicted.item()+1]



def save(model, save_dir, save_prefix, steps, best_acc=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
    if best_acc is not None:
      # 同時保存最高正確率
      with open(os.path.join(save_dir, 'best_accuracy.txt'), 'w') as f:
        f.write(str(best_acc))


