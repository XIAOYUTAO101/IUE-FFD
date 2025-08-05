import torch
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn.pytorch import GMMConv
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init

class GraphMetaPaths(nn.Module):
    def __init__(self, meta_paths):
        super(GraphMetaPaths, self).__init__()
        self.meta_paths = meta_paths

    def get_meta_path(self, g):
        _cached_graph = None
        _cached_coalesced_graph = []
        if _cached_graph is None or _cached_graph is not g:
            _cached_graph = g
            _cached_coalesced_graph.clear()
            for i,meta_path in enumerate(self.meta_paths):
                new_g = dgl.metapath_reachable_graph(
                    g, meta_path)
                new_g = dgl.to_homogeneous(new_g)
                new_g = dgl.to_simple_graph(new_g)
                new_g = dgl.add_self_loop(new_g)
                _cached_coalesced_graph.append(new_g)
        return _cached_coalesced_graph


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 dropout,
                 tail_activation=True,
                 activation=nn.ReLU(),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            modlist.append(activation)
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout))

            for _ in range(num_layers - 2):
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))

            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)



class LISeq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out, src = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            if isinstance(self.modlist[i], LILinear):
                out, src = self.modlist[i](out, src)
            elif isinstance(self.modlist[i], nn.Dropout):
                out = self.modlist[i](out)
        return out

class LILinear(nn.Module):
    def __init__(self, in_features, out_features, origin_infeat, bias = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LILinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features , in_features), **factory_kwargs))
        self.w_src = nn.Parameter(torch.empty((self.in_features, 1), **factory_kwargs))
        self.trans_src = nn.Linear(origin_infeat, in_features)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.fcs = nn.Linear(in_features * 3, out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_src, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, src):
        t_src = self.trans_src(src)  # (BS, in_dim)
        trans_input = input * t_src  # (BS, in_dim)
        out = F.linear(trans_input, self.weight, self.bias)  # (BS, 1, in_dim) (BS, in_dim, out_dim)
        # out =  self.fcs(trans_input)
        return out, src

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LIMLP(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 origin_infeat: int,
                 dropout: int,
                 tail_activation=True,
                 activation=nn.ReLU(),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None

        if num_layers == 1:
            modlist.append(LILinear(input_channels, output_channels, origin_infeat))
            if tail_activation:
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))
            self.seq = LISeq(modlist)
        else:

            modlist.append(LILinear(input_channels, hidden_channels, origin_infeat))


            for _ in range(num_layers - 2):
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))
                modlist.append(LILinear(hidden_channels, hidden_channels, origin_infeat))


            modlist.append(activation)
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout))
            modlist.append(LILinear(hidden_channels, output_channels, origin_infeat))

            if tail_activation:
                modlist.append(activation)
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout))
            self.seq = LISeq(modlist)
    def forward(self, x, h_self):
        return self.seq(x, h_self)



class GMPConv(nn.Module):
    def __init__(self,args, in_features, kernels_hid, out_features, mlp_activation = nn.ReLU(), activation=F.leaky_relu, feat_drop=0.0, device=None):
        super(GMPConv, self).__init__()
        self.out_features = out_features
        self.kernels_hid = kernels_hid
        self._n_kernels = args.n_kernels
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.device = device
        self.fc = nn.Linear(
            in_features, self._n_kernels *  self.kernels_hid, bias=False
        )
        self.fc2 = nn.Linear(
            in_features, self._n_kernels * self.kernels_hid, bias=False
        )
        self.fc3 = nn.Linear(
            self.kernels_hid * self._n_kernels, out_features, bias=False
        )
        self.fc_neigh_benign = LIMLP(input_channels= in_features, hidden_channels=out_features,
                                     output_channels=out_features,
                                     num_layers=1, origin_infeat= in_features, dropout=args.dropout, activation=mlp_activation)

        self.fc_neigh_fraud = LIMLP(input_channels= in_features, hidden_channels=out_features,
                                    output_channels=out_features,
                                    num_layers=1, origin_infeat= in_features, dropout=args.dropout, activation=mlp_activation)

        # self.w_fr            = nn.Parameter(torch.empty(self._in_src_feats, 1))
        self.fc_neigh = nn.Linear( in_features, out_features, bias=False)

        self.fc_balance = MLP(in_features, hidden_channels=out_features, output_channels=1,
                              dropout=args.dropout, num_layers=1)
        self.fc_self = nn.Linear(in_features, out_features, bias=True)

        self.fc_end = nn.Linear(in_features, out_features, bias=True)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_features, nhead=4,
                                       dim_feedforward=2 * in_features,
                                       dropout=args.dropout,
                                       batch_first=True),
            num_layers=1
        )

        self.E_group=nn.Parameter(torch.randn(4, in_features))

        self.balance_w = nn.Sigmoid()


        self.mu = nn.Parameter(torch.empty(self._n_kernels, self.kernels_hid))
        self.inv_sigma = nn.Parameter(torch.empty(self._n_kernels, self.kernels_hid))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)

    def agg_func(self,nodes):
        return {'neigh_fr': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label'] == 1).unsqueeze(-1)).sum(1),
                'neigh_be': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label'] == 0).unsqueeze(-1)).sum(1)
                }

    def mp_func(self,edges):
        src_fake_label = edges.src['label_unk']
        # src =  edges.edges()[0]
        src = edges.src['_ID']
        dst = edges.dst['_ID']

        return {'m': edges.src['h'], 'src': src, 'src_fake_label': src_fake_label}


    def mp_func1(self,edges):
        src_fake_label = edges.src['label_unk']
        # src =  edges.edges()[0]
        src = edges.src['_ID']
        dst = edges.dst['_ID']

        h_src = edges.src['h_']
        h_dst = edges.dst['h_']
        pseudo = h_dst - h_src
        inv_sigma = F.softplus(self.inv_sigma)
        gaussian = -0.5 * ((pseudo.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2 * inv_sigma.unsqueeze(0) ** 2).sum(-1)
        gaussian = torch.exp(gaussian).unsqueeze(-1)  # (E, K, 1)
        m = h_src.unsqueeze(1) * gaussian  # (E, K, D)
        m = m.view(-1, self._n_kernels * self.kernels_hid)

        return {'m': m, 'src': src, 'src_fake_label': src_fake_label}


    def agg_func1(self,nodes):
        return {
                'neigh_unk': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label'] == 2).unsqueeze(-1)).sum(1)}

    def forward(self, g,  feat, edge_weight=None , weights=None):
        with g.local_scope():
            g.srcdata['h'] = feat
            g.srcdata['h_'] = feat
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if g.is_block:
                    feat_dst = feat_src[: g.number_of_dst_nodes()]
                    g.dstdata['h_'] = self.fc2(feat_dst).view(
                -1, self._n_kernels, self.kernels_hid
            )

            # g.srcdata['hl'] = torch.cat((g.srcdata['h'], g.srcdata['label_unk'].unsqueeze(1)), dim=-1)
            msg_fn = fn.copy_u("h", "m")
            # msg_fn = fn.copy_u("hl", "m")

            if edge_weight is not None:
                assert edge_weight.shape[0] == g.num_edges()
                g.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Message Passing
            g.srcdata["h_"] = self.fc(feat_src).view(
                -1, self._n_kernels, self.kernels_hid
            )

            g.update_all(self.mp_func, self.agg_func)
            # g.update_all(msg_fn, fn.mean("m", "neigh"))
            neigh_fr = g.dstdata["neigh_fr"]
            neigh_be = g.dstdata["neigh_be"]


            g.update_all(self.mp_func1, self.agg_func1)
            neigh_unk = g.dstdata["neigh_unk"]
            # if not lin_before_mp:
            #     h_neigh = self.fc_neigh(h_neigh)
            neigh_fr = self.fc_neigh_fraud(neigh_fr, h_self)
            neigh_be = self.fc_neigh_benign(neigh_be, h_self)
            neigh_unk = self.fc3(neigh_unk)
            balance = self.balance_w(self.fc_balance(h_self))
            neigh_unk = balance * self.fc_neigh_fraud(neigh_unk, h_self) + (1 - balance) * self.fc_neigh_benign(
                neigh_unk, h_self)


            H = torch.stack([self.fc_self(h_self),neigh_be,neigh_fr,neigh_unk], dim=1)

            h_neigh = H + self.E_group.unsqueeze(0)  # (B, 4, D)


            H_out = self.encoder(h_neigh)  # (B, 4, D)
            rst = self.fc_end(H_out.mean(dim=1))  # (B, D)

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst

class IUE_GMP(nn.Module):
    def __init__(self, m, d, c, args, device):
        super(IUE_GMP, self).__init__()
        self.m = list(tuple(meta) for meta in m)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

        self.layers = args.layers
        self.convs = nn.ModuleList()
        self.relation_mlp = nn.ModuleList()
        for i in range(self.layers):
            self.convs.append(GMPConv(args, args.hidden_channels, args.kernels_hid, args.hidden_channels, device=device))
            self.relation_mlp.append(
                nn.Linear(args.hidden_channels * len(self.m) , args.hidden_channels))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.act_fn = nn.LeakyReLU()

        self.T = args.T
        self.distence_feature = nn.Parameter(torch.ones(2))
        self.dropout = args.dropout
        self.device = device

    def forward(self, graph):
        features = graph[0].srcdata['feat']
        h = self.act_fn(self.fcs[0](features))
        for layer in range(self.layers):
            h_list = []
            for i,rel in enumerate(graph[layer].etypes):
                h_rel = self.convs[layer](graph[layer][rel], h)
                # h_rel = torch.mean(h_rel, dim=1)
                h_list.append(h_rel)
            h = self.act_fn(self.relation_mlp[layer](torch.cat(h_list, dim=-1)))
        out = self.fcs[-1](h)
        return out,h


class IKT(object):
    def __init__(self, model):
        self.model = model
        self.history_importance = []
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.p_old = {self._canon(n): p.clone().detach()
                       for n, p in self.params.items()}
        self._precision_matrices = self.initialize_precision_matrices()
        self.normalized_uncertainty = {}

    @staticmethod
    def _canon(name: str) -> str:
        return name[7:] if name.startswith("module.") else name

    def initialize_precision_matrices(self):
        precisions = {}
        for n, p in self.params.items():
            precisions[self._canon(n)] = p.detach().clone().fill_(0)
        return precisions

    def calculate_importance(self, dataloader):
        new_importance = {}
        for n, p in self.params.items():
            new_importance[self._canon(n)] = p.detach().clone().fill_(0)
        self.model.eval()
        num_data = len(dataloader)
        for input_nodes, output_nodes, subgraph in dataloader:
            self.model.zero_grad()
            output, _ = self.model(subgraph)
            output = torch.sqrt(output.pow(2))
            loss = torch.sum(output, dim=1).mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                n = self._canon(n)
                if n in new_importance and p.grad is not None:
                    new_importance[n] += (p.grad ** 2) / num_data

        self.history_importance.append(new_importance)

        epsilon = 1e-6
        uncertainty = {}
        for n in new_importance:
            uncertainty[n] = (1 / (new_importance[n] + epsilon)).sum().item()

        all_uncertainties = list(uncertainty.values())
        min_uncertainty = min(all_uncertainties)
        max_uncertainty = max(all_uncertainties)
        range_ = max_uncertainty - min_uncertainty
        if range_ < 1e-8:
            normalized_uncertainty = {n: 0.5
                                      for n in uncertainty}
        else:
            normalized_uncertainty = {n: (σ - min_uncertainty) / (range_ + epsilon)
                                      for n, σ in uncertainty.items()}
        self.normalized_uncertainty = normalized_uncertainty

        self.unc_mean = float(np.mean(list(normalized_uncertainty.values())))

        m_max = 0.9
        m_min = 0.5
        dynamic_m = {}
        for n in normalized_uncertainty:
            dynamic_m[n] = m_max - (m_max - m_min) * normalized_uncertainty[n]

        for orig_n in self.params:
            n = self._canon(orig_n)
            self._precision_matrices[n] = dynamic_m[n] * self._precision_matrices[n] + (1 - dynamic_m[n]) * new_importance[n]

        print("Ω_max =", max(v.max().item() for v in self._precision_matrices.values()))

    def penalty(self, model):
        loss = 0
        epsilon = 1e-6
        for n, p in model.named_parameters():
            n = self._canon(n)
            if n in self._precision_matrices:
                prec  = self._precision_matrices[n]
                p_old = self.p_old[n]
                unc = self.normalized_uncertainty.get(n, self.unc_mean)
                factor = 1.0 / (unc + epsilon)
                loss += (factor * prec * (p - p_old).pow(2)).sum()
        return loss

    def get_history_importance(self):
        return self.history_importance

    def update_p_old(self):
        for orig_n, p in self.params.items():
            n = self._canon(orig_n)
            self.p_old[n] = p.clone().detach()


