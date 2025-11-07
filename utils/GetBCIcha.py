import os
import torch
from torch.utils import data
from scipy import io


# session 123 is training set, session4 is validation set, and session5 is testing set. 
def get_all_dataloader(args):
    dev = args.device
    # dev = torch.device("cuda")
    train = io.loadmat(os.path.join(args.data_path, f'Data_S{args.sub:02d}_Sess' + '.mat'))

    tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
    templabel = torch.Tensor(train['y_test']).view(-1)
    x_train = tempdata[:240]
    y_train = templabel[:240]

    x_valid = tempdata[180:240]
    y_valid = templabel[180:240]

    x_test = tempdata[240:340]
    y_test = templabel[240:340]

    # x_train = x_train.to(dev)
    # y_train = y_train.long().to(dev)
    # x_valid = x_valid.to(dev)
    # y_valid = y_valid.long().to(dev)
    # x_test = x_test.to(dev)
    # y_test = y_test.long().to(dev)

    x_train = x_train.to(dev)
    y_train = y_train.unsqueeze(1).float().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.unsqueeze(1).float().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.unsqueeze(1).float().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_dataset = data.TensorDataset(x_train, y_train)
    valid_dataset = data.TensorDataset(x_valid, y_valid)
    test_dataset = data.TensorDataset(x_test, y_test)

    trainloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        # pin_memory=True
    )
    validloader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=60,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )
    testloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )

    return trainloader, validloader, testloader