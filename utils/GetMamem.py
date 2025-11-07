import os
import torch
from torch.utils import data
from scipy import io


# session 123 is training set, session4 is validation set, and session5 is testing set. 
def get_all_dataloader(args):
    dev = torch.device("cpu")
    train = io.loadmat(os.path.join(args.data_path, 'U' + f'{args.sub:03d}' + '.mat'))
    temp_data = torch.Tensor(train['x_test']).unsqueeze(1)
    temp_label = torch.Tensor(train['y_test']).view(-1)

    # 划分训练集，验证集，测试集
    x_train = temp_data[:300]
    y_train = temp_label[:300]
    x_valid = temp_data[300:400]
    y_valid = temp_label[300:400]

    x_test = temp_data[400:500]
    y_test = temp_label[400:500]

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_dataset = data.TensorDataset(x_train, y_train)
    valid_dataset = data.TensorDataset(x_valid, y_valid)
    test_dataset = data.TensorDataset(x_test, y_test)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader
