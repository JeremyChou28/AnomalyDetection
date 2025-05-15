import os
import argparse
import time
import random
from torch.backends import cudnn
from utils.utils import *

from exp_anomaly_detection import Solver
import numpy as np
import torch

def str2bool(v):
    return v.lower() in ('true')

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


def main(config):
    seed_torch(config.seed)
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        solver.train()
        print('Training finished!')
        solver.test()
        print('Testing finished!')
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AD')
    parser.add_argument('--data_path', type=str, default='./processed_dataset/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    
    parser.add_argument('--model_name', type=str, default='Ours')
    parser.add_argument('--win_size', type=int, default=84)
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test', choices=['scratch','train', 'test'])
    parser.add_argument('--seed', type=int, default=2023)


    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    start_time = time.time()
    main(config)
    print("Spend Time: ", time.time() - start_time)
