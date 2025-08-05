import numpy as np
import torch
import os, time
import os.path as osp
import argparse
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def parse_args():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description='Node CLF')

    parser.add_argument('--mode', type=str,default='dynamic', help='dynamic or static')

    # training
    parser.add_argument('--lr', type=float, default=5e-3,help='[1e-2, 5e-3, 1e-3]')
    parser.add_argument('--re_lr', type=float, default=5e-3,help='[5e-3, 1e-3, 1e-3]')
    parser.add_argument('--epochs', type=int, default=300, help='[300, 300, 300]')
    parser.add_argument('--repochs', type=int, default=300)


    parser.add_argument('--alpha', type=float, default=0.5, help='[0.5, 0.7, 0.5]')
    parser.add_argument('--beta', type=float, default=1.0, help= "[1.0, 1.5, 1.0]")
    parser.add_argument('--sigma', type=float, default=0.001, help='')

    parser.add_argument('--layers', type=int, default=1, help='')
    parser.add_argument('--n_kernels', type=int, default=3, help='[2,3,3]')
    parser.add_argument('--hidden_channels', type=int, default=64, help= "[64, 64, 64]")
    parser.add_argument('--kernels_hid', type=int, default=64, help= "[64, 64, 64]")

    parser.add_argument('--datasets', type=str, default='CCT', help='datasets:[CCT,Vesta,Amazon]')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--dropout', type=float, default=0., help='[0.,0.,0.4]')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--T', type=float, default=1.5, help='[1.5,1.5,2.0]')
    parser.add_argument('--seed', type=int, default=123)


    args = parser.parse_args()

    return args