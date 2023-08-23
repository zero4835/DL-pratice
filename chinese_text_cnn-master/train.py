import os
import sys
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )  # 根据需要调整 step_size 和 gamma
    steps = 0
    best_acc = 0
    last_step = 0

    dev_acc_history = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            with torch.no_grad():
                feature.t_(), target.sub_(1)

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (
                    torch.max(logits, 1)[1].view(target.size()).data == target.data
                ).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    "\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})".format(
                        steps, loss.item(), train_acc, corrects, batch.batch_size
                    )
                )
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                dev_acc_history.append(dev_acc)

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print("Saving best model, acc: {:.4f}%\n".format(best_acc))
                        save(model, args.save_dir, "best", steps)

                        plt.plot(
                            range(args.test_interval, steps + 1, args.test_interval),
                            dev_acc_history,
                            label="dev_acc",
                        )
                        plt.xlabel("Steps")
                        plt.ylabel("Accuracy")
                        plt.title("Dev Accuracy During Training")
                        plt.legend()
                        plt.savefig("./data/dev_accuracy_plot.png")  # 保存为 PNG 文件
                        # plt.show()
                else:
                    if steps - last_step >= args.early_stopping:
                        print(
                            "\nearly stop by {} steps, acc: {:.4f}%".format(
                                args.early_stopping, best_acc
                            )
                        )
                        plt.plot(
                            range(args.test_interval, steps + 1, args.test_interval),
                            dev_acc_history,
                            label="dev_acc",
                        )
                        plt.xlabel("Steps")
                        plt.ylabel("Accuracy")
                        plt.title("Dev Accuracy During Training")
                        plt.legend()
                        plt.savefig("./data/dev_accuracy_plot.png")  # 保存为 PNG 文件
                        # plt.show()
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print(
        "\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(
            avg_loss, accuracy, corrects, size
        )
    )
    return float("{:.4f}".format(accuracy.item()))


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = "{}_steps_{}.pt".format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
