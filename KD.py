from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
import numpy as np
from scipy.linalg import inv

device ="cuda"
# def str_distill(old_emb_list,emb_list,weight,gamma):
#     losses=[]
#     loss = torch.tensor(0.0).to(device)
#     for i in range(len(emb_list)):
#         loss =loss+ torch.sqrt(F.mse_loss(emb_list[i], old_emb_list[i]))
#         losses.append(loss)
#     weight_softmax= F.softmax(weight, dim=-1)
#     structural_loss = gamma * torch.stack(losses, dim=0).dot(weight_softmax)
#
#     return structural_loss


class KT(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(KT, self).__init__()
        self.T = T

    def forward(self,y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

class Structural_Distillation(nn.Module):
    def __init__(self,t,T,delta):
        super(Structural_Distillation, self).__init__()
        self.t = t
        self.T = T
        self.graphs_weight = nn.Parameter(torch.linspace(start=self.t, end=1, steps=self.t), requires_grad=True)
        self.distence_feature = nn.Parameter(torch.ones(2),requires_grad=True)
        self.huber_loss=nn.HuberLoss(delta=delta,reduction='sum')

    def forward(self,param_old,model,encoder,old_encoder):
        param_new=[]
        losses=[]
        for name, param in model.named_parameters():
            name = name.replace('.', '_')
            if "attention_weights" in name:
                continue
            param_new.append(param.data.clone())
        for i in range(self.t):
            loss=torch.tensor(0.0).to(device)
            for j in range(len(param_new)):
                param_new_j = torch.tensor(param_new[j]) if not isinstance(param_new[j], torch.Tensor) else param_new[j]
                param_old_i_j = torch.tensor(param_old[i][j]) if not isinstance(param_old[i][j], torch.Tensor) else param_old[i][j]
                loss=loss+self.huber_loss(param_new_j,param_old_i_j)
            losses.append(loss)
        weight_softmax=F.softmax(self.graphs_weight,dim=-1)
        structural_loss = torch.stack(losses, dim=0).dot(weight_softmax)

        euclidean_feature = torch.norm(encoder - old_encoder,dim=1).mean()
        cosine_feature = 1 - F.cosine_similarity(encoder, old_encoder, dim=1).mean()
        features_weight=F.softmax(self.distence_feature/self.T, dim=-1)
        feature_loss = features_weight[0] * euclidean_feature + features_weight[1] * cosine_feature

        return structural_loss,feature_loss


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs


def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    sim_i_j = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)

    sim_i_i = torch.exp(torch.mm(z_i, z_i.t()) / temperature)
    sim_j_j = torch.exp(torch.mm(z_j, z_j.t()) / temperature)

    sim_i_i.fill_diagonal_(0)
    sim_j_j.fill_diagonal_(0)

    loss_i = -torch.log(sim_i_j / (sim_i_i.sum(1) + sim_i_j))
    loss_j = -torch.log(sim_i_j / (sim_j_j.sum(1) + sim_i_j))

    return (loss_i + loss_j).mean()

def cosine_similarity_loss(z_i, z_j,beta=5):
    cos_sim = F.cosine_similarity(z_i, z_j, dim=-1)
    cos_distance = 1 - cos_sim
    loss =beta * cos_distance.mean()
    return loss

def structural_distillation_loss(z_i, z_j, beta=250):
    str_loss=F.mse_loss(z_i, z_j)
    loss = beta * str_loss
    return loss

def manhattan_distance_loss(z_i, z_j, beta=0.1):
    manhattan_distance = torch.sum(torch.abs(z_i - z_j), dim=-1)
    loss = beta* manhattan_distance.mean()
    return loss


