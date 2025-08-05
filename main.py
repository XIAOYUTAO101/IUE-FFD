import copy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import warnings
import pandas as pd
import torch
from datetime import datetime
from args import parse_args
from GraphUpdate import Graph_Update,make_subgraph
import time

warnings.filterwarnings('ignore')

from utils import *
from model import *
from KD import KT

device = "cuda:0"
device1 = "cpu"


def main(args,graphs):
    input_dim = graphs['features'].shape[1]
    if args.datasets == "CCTFD":
        meta_paths = [['tc', 'ct'], ['th', 'ht'], ['tm', 'mt'], ['tc2', 'ct2']]
    elif args.datasets == "Vesta":
        meta_paths = [['tu', 'ut'], ['tc', 'ct'],['th', 'ht']]
    elif args.datasets == "Amazon":
        meta_paths = [['rp', 'pr'], ['ru', 'ur'], ['rh', 'hr']]
    GraphGenerator = GraphMetaPaths(meta_paths)
    model = IUE_GMP(m=meta_paths,
                    d=input_dim,
                    c=args.class_num,
                    args=args,
                    device=device).to(device)

    model_mas = IKT(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fcn = torch.nn.functional.cross_entropy
    folder_path = os.path.join("models", args.datasets)
    os.makedirs(folder_path, exist_ok=True)
    save_path = f"models/{args.datasets}/{args.datasets}_bestmodel.pth"
    best_score = 0
    best_loss=100000
    cur_step = 0

    ori_graph = graphs['graph']
    features = graphs['features']
    labels = graphs['labels']
    train_mask = graphs['train']
    val_mask = graphs['val']
    test_mask = graphs['test']
    g = Modifiy_graph(GraphGenerator,ori_graph,features,labels,train_mask)
    train_loader, val_loader, test_loader = split_data(args, g, train_mask, val_mask, test_mask)

    scheduler = LinearSchedule(optimizer, args.epochs, base_lr=args.lr)
    n_per_cls = [(labels[train_mask] == i).sum() for i in range(args.class_num)]
    loss_w_train = [1. / max(i, 1) for i in n_per_cls]
    loss_w_train = torch.tensor(loss_w_train).to(device)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for input_nodes, output_nodes, subgraph in train_loader:
            subgraph = [g.to(device) for g in subgraph]
            labels = subgraph[-1].dstdata['label']
            optimizer.zero_grad()
            # optimizer_ditill.zero_grad()
            logits,h = model(subgraph)
            loss = loss_fcn(logits, labels,weight=loss_w_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        loss_val, model, best_marco_f1, best_marco_f1_thr, best_recall  = evaluate_val(args, loss_fcn, model, val_loader)

        if epoch % 5 == 0:
            f1, recall, fpr, auc = evaluate_test(
                best_marco_f1_thr, model, train_loader)
            avg_train_loss = total_loss / num_batches
            print('| Epoch {:3d} | Train: loss={:.3f}, train_f1={:5.1f}% | Val:loss={:.3f}, val_f1={:5.1f}%'.format(
                epoch + 1, avg_train_loss, 100 * f1, loss_val, 100 * best_marco_f1), end='\r')

            if (loss_val + args.sigma) < best_loss or best_marco_f1 > best_score:
                best_score = best_marco_f1
                best_loss = loss_val
                cur_step = 0
                save_checkpoint(model, save_path)
            else:
                cur_step += 1
                if cur_step == 5:
                    print('| Epoch {:3d} | Val: best_loss={:.3f}, best_f1_score={:5.1f}% |'.format(
                        epoch + 1, best_loss, 100 * best_score))
                    break
            print('| Epoch {:3d} | Train: loss={:.3f}, train_f1={:5.1f}% | Val:loss={:.3f}, Val_f1={:5.1f}% |'.format(
            epoch + 1, avg_train_loss, 100 * f1, loss_val, 100 * best_marco_f1))
        scheduler.step()
    mf1, recall, Fpr, auc = evaluate_test(best_marco_f1_thr, model, test_loader)
    print("Test F1:{:5.2f} | Recall:{:5.2f} | Fpr:{:5.2f} | AUC:{:5.2f}".format(
        100 * mf1, 100 * recall, 100 * Fpr, 100 * auc))
    model_mas.calculate_importance(train_loader)
    model_mas.update_p_old()
    return  model, model_mas,loss_fcn, GraphGenerator


def Refine(args, model, model_mas, graphs, new_data, GraphGenerator, loss_func):
    best_loss = 100000
    best_score = 0
    cur_step = 0
    model.to(device)
    KD_loss = KT(args.T)

    best_old_model = copy.deepcopy(model)

    labels = new_data['labels']
    train_mask = new_data['train']
    val_mask = new_data['val']
    test_mask = new_data['test']

    train_loader, val_loader, test_loader = split_data(args, graphs, train_mask, val_mask, test_mask)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.re_lr, weight_decay=args.weight_decay)
    scheduler = LinearSchedule(optimizer, args.repochs, base_lr=args.re_lr)

    n_per_cls = [(labels[train_mask] == i).sum() for i in range(args.class_num)]
    train_loss_w = [1. / max(i, 1) for i in n_per_cls]
    train_loss_w = torch.tensor(train_loss_w).to(device)

    n_per_cls_val = [(labels[val_mask] == i).sum() for i in range(args.class_num)]
    val_loss_w = [1. / max(i, 1) for i in n_per_cls_val]
    val_loss_w = torch.tensor(val_loss_w).to(device)

    for epoch in range(args.repochs):
        model.train()
        best_old_model.eval()
        total_loss = 0
        total_loss_mas = 0
        total_loss_div = 0
        num_batches = 0
        for input_nodes, output_nodes, subgraph in train_loader:
            subgraph = [g.to(device) for g in subgraph]
            labels = subgraph[-1].dstdata['label']
            optimizer.zero_grad()
            logits, h = model(subgraph)
            old_logits, old_h = best_old_model(subgraph)
            loss_div = KD_loss(h, old_h)
            loss_mas = model_mas.penalty(model)
            loss = loss_func(logits, labels, weight=train_loss_w)
            loss = loss+ args.alpha * loss_mas + args.beta * loss_div

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_mas += loss_mas.item()
            total_loss_div += loss_div.item()
            num_batches += 1
        loss_val, model, best_marco_f1, best_marco_f1_thr, best_recall  = evaluate_val(args, loss_func, model, val_loader)
        if epoch % 5 == 0:
            f1, recall, fpr, auc = evaluate_test(
                best_marco_f1_thr, model, train_loader)
            avg_train_loss = total_loss / num_batches
            avg_train_mas_loss = total_loss_mas / num_batches
            avg_train_div_loss = total_loss_div / num_batches
            print('| Epoch {:3d} | Train: loss={:.3f}, mas_loss={:.3f}, div_loss={:.3f}, train_f1={:5.1f}% | Val:loss={:.3f}, val_f1={:5.1f}%'.format(
                epoch + 1, avg_train_loss,avg_train_mas_loss, avg_train_div_loss, 100 * f1, loss_val, 100 * best_marco_f1), end='\r')

            if (loss_val + args.sigma) < best_loss or best_marco_f1 > best_score:
                best_score = best_marco_f1
                best_loss = loss_val
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == 5:
                    print('| Epoch {:3d} | Val: best_loss={:.3f}, best_f1_score={:5.1f}% |'.format(
                        epoch + 1, best_loss, 100 * best_score))
                    break
            print('| Epoch {:3d} | Train: loss={:.3f}, mas_loss={:.3f}, div_loss={:.3f}, train_f1={:5.1f}% | Val:loss={:.3f}, Val_f1={:5.1f}% |'.format(
                epoch + 1, avg_train_loss, avg_train_mas_loss, avg_train_div_loss, 100 * f1, loss_val, 100 * best_marco_f1))
        scheduler.step()

    f1, recall, fpr, auc = evaluate_test(best_marco_f1_thr, model, test_loader)
    print("Test F1:{:5.2f} | Recall:{:5.2f} | Fpr:{:5.2f} | AUC:{:5.2f}".format(
        100 * f1, 100 * recall, 100 * fpr, 100 * auc))
    model_mas.calculate_importance(train_loader)
    model_mas.update_p_old()
    return model, model_mas


def Run(args, graphs, test_data, node_types):
    best_model, model_mas, loss_func, GraphGenerator= main(args, graphs)
    best_model = best_model.to(device)
    GraphGenerator = GraphGenerator.to(device)
    all_tasks = []
    f1, recall, fpr, auc=0,0,0,0
    for slot in test_data:
        new_graphs_id = {}
        print(f"Processing transactions for time slot {slot}")
        new_graphs_id['trans_id']=test_data[slot]['trans_id']
        graphs = Graph_Update(args, graphs, test_data[slot], node_types)
        new_g = graphs['graph'].to(device)
        if args.datasets == 'CCTFD':
            new_graphs_id['client_id'] = test_data[slot]['client_id']
            new_graphs_id['merchant_id'] = test_data[slot]['merchant_id']
            new_graphs_id['card_id'] = test_data[slot]['card_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']
        elif args.datasets == 'Vesta':
            new_graphs_id['user_id'] = test_data[slot]['user_id']
            new_graphs_id['card_id'] = test_data[slot]['card_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']
        elif args.datasets == 'Amazon':
            new_graphs_id['asin'] = test_data[slot]['asin']
            new_graphs_id['user_id'] = test_data[slot]['user_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']

        subgraph_hetero = make_subgraph(args,new_g, new_graphs_id)
        hetero_graphs = Modifiy_graph(GraphGenerator, subgraph_hetero, test_data[slot]['features'],
                                      test_data[slot]['labels'], test_data[slot]['train'])
        print("Refining model...")
        best_model, model_mas = Refine(args, best_model, model_mas, hetero_graphs, test_data[slot], GraphGenerator, loss_func)
        print("Refining model finished.")
        all_tasks.append(new_graphs_id)


        ori_g = graphs['graph'].to(device1)
        features = graphs['features'].to(device1)
        labels = graphs['labels'].to(device1)
        train_mask = graphs['train'].to(device1)
        val_mask = graphs['val'].to(device1)
        test_mask = graphs['test'].to(device1)
        g_all = Modifiy_graph(GraphGenerator, ori_g, features, labels, train_mask)
        _,val_mask,test_loader = split_data(args, g_all, train_mask, val_mask, test_mask)
        _ ,_, _, best_marco_f1_thr, _ = evaluate_val(args, loss_func, best_model, val_mask)
        f1,recall,fpr,auc = evaluate_test(best_marco_f1_thr,best_model,test_loader)
        print("Finally F1:{:5.2f}% , Recall:{:5.2f}% , FPR:{:5.2f}% , AUC:{:5.2f}%".format(100 * f1, 100 * recall,100 * fpr, 100 * auc))

    return f1,recall,fpr,auc



if __name__ == '__main__':
    args = parse_args()
    # set_seed()
    graphs,test_data,path = load_graph(args.datasets)
    node_types = load_nodes_types(args.datasets)
    print(f"Loaded {args.datasets} dataset.")
    print(f"Dataset path: {path}")
    print(f"Features shape: {graphs['features'].shape}")
    f1_, recall_, fpr_, auc_ = Run(args, graphs, test_data, node_types)

