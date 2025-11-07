import sys
import os
import torch
from utils.functions import train_network, save_res, setup_seed
from corAtt.CorAtt import CorrAttMamem
from utils.GetMamem import get_all_dataloader
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./conf/", config_name="mamem.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    repeat = 10
    setup_seed(3407)
    res = torch.zeros(11, repeat)
    for i in range(11):
        for j in range(repeat):
            ap = argparse.ArgumentParser()
            ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
            ap.add_argument('--sub', type=int, default=1 + i, help='subjectxx you want to triain')
            ap.add_argument('--lr', type=float, default=cfg.lr, help='learning rate')
            ap.add_argument('--wd', type=float, default=cfg.wd, help='weight decay')
            ap.add_argument('--iterations', type=int, default=180, help='number of training iterations')
            ap.add_argument('--epochs', type=int, default=cfg.epochs,
                            help='number of epochs that you want to use for split EEG signals')
            ap.add_argument('--bs', type=int, default=64, help='batch size')
            ap.add_argument('--model_path', type=str, default='./checkpoint/mamem/',
                            help='the folder path for saving the model')
            ap.add_argument('--data_path', type=str, default='/home/huchen/data/EEGdata/EEGdata/MAMEM/', help='data path')
            ap.add_argument('--res_path', type=str, default='./result/mamem/',
                            help='data path')
            ap.add_argument("--scoring_metric", choices=["acc", "auc"], default="acc", help="performance_metric")
            ap.add_argument('--optim', type=str, default=cfg.optim, help='Optimization method.')
            ap.add_argument('--device', type=str, default='cuda:1', help='device')
            ap.add_argument('--in_size', type=int, default=cfg.in_size)
            ap.add_argument('--out_size', type=int, default=cfg.in_size)
            ap.add_argument('--metric', type=str, default=cfg.metric)
            ap.add_argument('--nclass', type=int, default=5)
            args, unknown = ap.parse_known_args()

            train_loader, valid_loader, test_loader = get_all_dataloader(args)
            net = CorrAttMamem(args)
            if not os.path.exists(args.res_path):
                os.makedirs(args.res_path)
            final_path = os.path.join(args.res_path,
                                      f'in_size{args.in_size}_out_size{args.out_size}_epochs{args.epochs}'
                                      f'_lr{args.lr}_wd{args.wd}_{args.optim}_metric{cfg.metric}.txt')

            acc = train_network(net, train_loader, valid_loader, test_loader, args)

            print(f'{acc * 100:.2f}')
            res[i, j] = acc * 100
            print(f'{acc * 100:.2f}')
            print(acc)
            print(res)
            save_res(res, final_path)

if __name__ == '__main__':
    main()