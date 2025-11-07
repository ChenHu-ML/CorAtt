import sys
import os
import torch.nn as nn
from sklearn.metrics import roc_auc_score as ras
import numpy as np
from torch.optim import RAdam, NAdam, Adadelta, Adam, SGD
import torch
import random
sys.path.append("..")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def select_optimizer(model_parameters, optimizer_name, lr=1e-3, weight_decay=0):
    optimizer_name = optimizer_name.lower()
    # print(optimizer_name)
    model_parameters = [p for p in model_parameters if p.requires_grad]
    if optimizer_name == 'radam' and RAdam is not None:
        return RAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adadelta':
        return Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'nadam' and NAdam is not None:
        return NAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd' and SGD is not None:
        return SGD(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}.")


def generate_model_path(args):
    filename = (f'repeat{args.repeat}_sub{args.sub}_bs{args.bs}_epochs{args.epochs}_lr{args.lr}_wd{args.wd}_'
                f'in_size{args.in_size}_out_size{args.out_size}_{args.metric}.pt')

    return filename


def test_network_acc(net, test_loader):
    net.eval()
    acc, test_len = 0, 0
    for xb, yb in test_loader:
        with torch.no_grad():
            test_len += yb.shape[0]
            pred = net(xb)
            yb = yb.to(pred.device)
            acc += (torch.max(pred, 1).indices == yb).sum().item()
            # if torch.argmax(softmax(pred)) == yb:
            #     acc += 1

    return acc / test_len


def test_network_auc(net, test_loader):
    net.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = net(xb).cpu()
            prob = torch.sigmoid(pred.squeeze(-1))
            y_pred.append(prob.cpu())
            y_true.append(yb.cpu())
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    return ras(y_true.detach().numpy(), y_pred.detach().numpy())



def save_res(res, res_path):
    res = res.numpy()
    mean = np.mean(res).item()
    std = np.std(res, axis=1, ddof=0).mean()
    mean_st = [f'{x:.4f}' for x in np.mean(res, axis=1).tolist()]
    std_st = [f'{x:.4f}' for x in np.std(res, axis=1, ddof=0).tolist()]
    print(f"mean:{mean:.2f}\tstd:{std:.2f}")
    header_info = 'Mean: {:.4f}\t'.format(mean) + 'Std: {:.4f}\n'.format(std) \
                  + f'St.Mean: {", ".join(mean_st)}\n' + f'St.Std:  {", ".join(std_st)}\n'

    np.savetxt(res_path, res, fmt='%.4f', comments='', delimiter='\t', header=header_info)


def train_network(net, train_loader, valid_loader, test_loader, args):

    criterion = nn.CrossEntropyLoss()

    optimizer = select_optimizer(net.parameters(), args.optim, lr=args.lr, weight_decay=args.wd)

    best_loss = 1e10
    test_metric = 0
    for ite in range(args.iterations):

        net.train()
        val_acc, train_acc, train_len, val_len, val_loss, train_loss = 0, 0, 0, 0, 0, 0
        for xb, yb in train_loader:
            train_len += yb.shape[0]
            out = net(xb)
            yb = yb.to(out.device)
            loss = criterion(out, yb)
            optimizer.zero_grad()

            pred_labels = out.argmax(dim=1)
            train_acc += (pred_labels == yb).sum().item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * yb.shape[0]
        net.eval()

        with torch.no_grad():
            for xb, yb in valid_loader:
                val_len += yb.shape[0]
                out = net(xb)
                yb = yb.to(out.device)
                pred_labels = out.argmax(dim=1)
                val_acc += (pred_labels == yb).sum().item()

                val_loss += criterion(out, yb) * yb.shape[0]

        if val_loss < best_loss:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            best_loss = val_loss
            final_path = generate_model_path(args)
            final_path = os.path.join(args.model_path, final_path)
            torch.save(net, final_path)
            testnet = torch.load(final_path)
            if args.scoring_metric == 'acc':
                test_metric = test_network_acc(testnet, test_loader)
            elif args.scoring_metric == 'auc':
                test_metric = test_network_auc(testnet, test_loader)
        if ite == 0 or ite % 1 == 0 or ite == args.iterations - 1:
            print(
                f'epoch:{ite + 1:03d}/{args.iterations} '
                f'train_loss:{train_loss / train_len:.4f} train_acc:{train_acc / train_len:.4f} '
                f'val_loss:{val_loss / val_len:.4f} val_acc:{val_acc / val_len:.4f} '
                f'test_{args.scoring_metric}:{test_metric:.4f}')

    return test_metric
