import torch
from utils.functions import train_network, save_res
from corAtt.CorAtt import CorrAttCha
from utils.GetBCIcha import get_all_dataloader
import os
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./conf/", config_name="bcicha.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    name = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    repeat = 3
    res = torch.zeros(16, repeat)
    for i in range(16):
        # torch.manual_seed(cfg.seed)
        # torch.cuda.manual_seed(cfg.seed)
        # torch.cuda.manual_seed_all(cfg.seed)
        for j in range(repeat):
            ap = argparse.ArgumentParser()
            ap.add_argument('--repeat', type=int, default=cfg.ite, help='No.xxx repeat for training model')
            ap.add_argument('--sub', type=int, default=name[i], help='subjectxx you want to train')
            ap.add_argument('--lr', type=float, default=cfg.lr, help='learning rate')
            ap.add_argument('--wd', type=float, default=cfg.wd, help='weight decay')
            ap.add_argument('--iterations', type=int, default=cfg.ite, help='number of training iterations')
            ap.add_argument('--epochs', type=int, default=cfg.epochs, help='number of epochs that you want to use for split EEG signals')
            ap.add_argument('--bs', type=int, default=cfg.bs, help='batch size')
            ap.add_argument('--model_path', type=str, default='./checkpoint/plot/BCIcha/', help='the folder path for saving the model')
            ap.add_argument('--data_path', type=str, default='/home/huchen/data/EEGdata/EEGdata/BCIcha/', help='data path')
            ap.add_argument('--res_path', type=str, default='./result/BCIcha/',
                            help='data path')
            ap.add_argument("--scoring_metric", choices=["acc", "auc"], default="auc", help="performance_metric")
            ap.add_argument('--optim', type=str, default=cfg.optim, help='Optimization method.')
            ap.add_argument('--device', type=str, default=cfg.device, help='device')
            ap.add_argument('--in_size', type=int, default=cfg.in_size)
            ap.add_argument('--out_size', type=int, default=cfg.in_size)
            ap.add_argument('--metric', type=str, default=cfg.metric)
            ap.add_argument('--power', type=float, default=cfg.power)
            ap.add_argument('--nclass', type=int, default=1)

            args, unknown = ap.parse_known_args()
            # args.model_path = cfg.path

            train_loader, valid_loader, test_loader = get_all_dataloader(args)

            net = CorrAttCha(args)
            if not os.path.exists(args.res_path):
                os.makedirs(args.res_path)
            final_path = os.path.join(args.res_path,
                                      f'ite_{cfg.ite}_bs{cfg.bs}in_size{args.in_size}_out_size{args.out_size}_epochs{args.epochs}'
                                      f'_lr{args.lr}_wd{args.wd}_{args.optim}_power{cfg.power}_metric{cfg.metric}.txt')

            auc = train_network(net, train_loader, valid_loader, test_loader, args)
            print(f'{auc * 100:.2f}')
            res[i, j] = auc * 100
            print(f'{auc * 100:.2f}')
            print(auc)
            print(res)
            save_res(res, final_path)


if __name__=='__main__':
    main()
