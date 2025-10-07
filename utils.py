import numpy
from dgl.dataloading import NeighborSampler, DataLoader
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score
import torch
import pickle
import numpy as np

import random
import os
import math
import dgl
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import f1_score, precision_score, roc_auc_score, confusion_matrix
from warnings import warn
from KD import DistillKL
device0 ="cuda"
device1 = "cpu"

# Function to evaluate the model on the validation set and find the best threshold.
def evaluate_val(global_predictions, loss_fuc, model, test_loader):
    model.eval()
    with torch.no_grad():
        best_model, best_marco_f1, best_marco_f1_thr, best_recall = None, 0, 0, 0
        total_loss = 0.0
        total_preds = []
        total_labels = []

        for input_nodes, output_nodes, subgraph in test_loader:
            labels = subgraph[-1].dstdata['label']
            output,_ = model(subgraph)
            loss = loss_fuc(output, labels)
            total_loss += loss.item()


            probs = output.softmax(1).cpu().numpy()
            total_preds.extend(probs[:, 1])
            total_labels.extend(labels.cpu().numpy())


        for thres in np.linspace(0.05, 0.95, 19):
            preds = (np.array(total_preds) > thres).astype(int)
            mf1 = f1_score(total_labels, preds, average='macro')
            recall = recall_score(total_labels, preds, average='macro')

            if mf1 > best_marco_f1:
                best_marco_f1 = mf1
                best_marco_f1_thr = thres
                best_recall = recall
                best_model = model

    return total_loss / len(
        test_loader), best_model, best_marco_f1, best_marco_f1_thr, best_recall

# Function to evaluate the model on the test set.
def evaluate_test(thres_f1, model, test_loader):
    model = model.to(device0)
    model.eval()
    with torch.no_grad():
        total_preds = []
        total_labels = []
        for input_nodes,output_nodes,subgraph in test_loader:
            # features = subgraph.ndata['x']
            labels = subgraph[-1].dstdata['label']

            output,_  = model(subgraph)
            prediction = output.softmax(1).cpu().numpy()[:,1]
            probs = output.softmax(1).cpu().numpy()
            total_preds.extend(probs[:, 1])
            total_labels.extend(labels.cpu().numpy())
        preds = (np.array(total_preds) > thres_f1).astype(int)
        mf1 = f1_score(total_labels, preds, average='macro')
        recall = recall_score(total_labels, preds, average='macro')
        fprs = []
        conf_matrix = confusion_matrix(total_labels, preds)
        for j in range(conf_matrix.shape[0]):
            fp = conf_matrix[:, j].sum() - conf_matrix[j, j]
            tn = conf_matrix.sum() - (conf_matrix[j, :].sum() + conf_matrix[:, j].sum() - conf_matrix[j, j])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs.append(fpr)
        Fpr = np.mean(fprs)
        # best_marco_f1, best_marco_f1_thr, best_recall = get_max_macrof1_recall(labels, prediction)
        auc = roc_auc_score(total_labels, total_preds)
        return mf1, recall, Fpr, auc




def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

# Function to load graph data.
def load_graph(dataset):
    if dataset == "CCT":
        path = 'data/graphs_CCT_online.pkl'
        path_2 = 'data/test_data_CCT_online.pkl'
        with open(path, 'rb') as f:
            graphs = pickle.load(f)
        with open(path_2, 'rb') as f:
            test_data = pickle.load(f)
        return graphs, test_data, path

    elif dataset == "Vesta":
        path = 'data/graphs_Vesta_online.pkl'
        path_2 = 'data/test_data_Vesta_online.pkl'
        with open(path, 'rb') as f:
             graphs =pickle.load(f)
        with open(path_2, 'rb') as f:
            test_data = pickle.load(f)
        return graphs, test_data, path

    elif dataset == "Amazon":
        path = 'data/graphs_Amazon_online.pkl'
        path_2 = 'data/test_data_Amazon_online.pkl'
        with open(path, 'rb') as f:
            graphs = pickle.load(f)
        with open(path_2, 'rb') as f:
            test_data = pickle.load(f)
        return graphs,test_data,path

# Function to load node type information.
def load_nodes_types(dataset):
    if dataset == "CCT":
        with open('data/nodes_types_CCT.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types
    elif dataset == "Vesta":
        with open('data/nodes_types_Vesta.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types
    elif dataset == "Amazon":
        with open('data/nodes_types_Amazon.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types

# Function to modify the graph structure for the model.
def Modifiy_graph(GraphGenerator, graph, features, labels, idx):
    graph = graph.to(device1)
    idx = idx.to(device1)
    labels = labels.to(device1)
    features = features.to(device1)
    g = GraphGenerator.get_meta_path(graph)
    num_nodes = g[0].number_of_nodes()

    selected_indices = torch.nonzero(idx).squeeze()
    label_unk = (torch.ones(num_nodes) * 2).long()
    label_unk[selected_indices] = labels[selected_indices]

    edge_types = [('node', f'relation{i}', 'node') for i in range(1, len(g) + 1)]

    all_edges = [g_.edges() for g_ in g]

    hg = dgl.heterograph(dict(zip(edge_types, all_edges)))
    hg.nodes['node'].data['feat']  = features.contiguous()
    hg.nodes['node'].data['label'] = labels
    hg.nodes['node'].data['label_unk'] = label_unk

    return hg

# Function to split data into training, validation, and test loaders.
def split_data(args, data, train_mask, val_mask, test_mask):
    n_sample = {}
    for e in data.etypes:
        n_sample[e] = -1
    n_samples = [n_sample] * args.layers
    sampler = NeighborSampler(n_samples)
    idx_train = torch.nonzero(train_mask).squeeze()
    idx_valid = torch.nonzero(val_mask).squeeze()
    idx_test = torch.nonzero(test_mask).squeeze()
    train_loader = DataLoader(data, idx_train, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              use_uva=True)
    valid_loader = DataLoader(data, idx_valid, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              use_uva=True)
    test_loader = DataLoader(data, idx_test, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             use_uva=True)
    return train_loader, valid_loader, test_loader



def save_checkpoint(model, save_path):
    """Saves model when validation loss decreases."""
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, save_path):
    """Load the latest checkpoint."""
    model.load_state_dict(torch.load(save_path))
    return model

def set_seed(args=None):
    seed = 1 if not args else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def setup(args):
    set_random_seed(args['seed'])
    args['num_heads'] = [args['num_heads'] for _ in range(args['layer'])]
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def has_one(tensor):
    return (tensor == 1).nonzero().size[0] > 0

# A class for linear learning rate schedule with warmup and decay.
class LinearSchedule(lrs.LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to base_lr over `warmup_steps` training steps.
    Linearly decreases learning rate from base_lr to 0 over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, t_total, base_lr, warmup_steps=0, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps)) * self.base_lr
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps))) * self.base_lr


