import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
import copy
from torch.nn import BatchNorm1d
from collections import OrderedDict
import math


class TDrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(TDrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)    #5000,64
        self.conv2 = GCNConv(args.hidden_features, args.output_features)  #5000,64,64
        self.device = args.device

        self.num_features_list = [args.hidden_features * r for r in [1]]

        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list

        self.sim_network = th.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]
        # self.W_mean = th.nn.Sequential(creat_network(mod_self, 'W_mean'))
        # self.W_bias = th.nn.Sequential(creat_network(mod_self, 'W_bias'))
        # self.B_mean = th.nn.Sequential(creat_network(mod_self, 'B_mean'))
        # self.B_bias = th.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        # self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        # self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(args.hidden_features)   #64


    def forward(self, data,edge_weight,edge_infer):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if edge_infer:
            edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_pred = edge_weight
    #rootindex feature pour out
        # rootindex = data.rootindex
        # root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        # batch_size = max(data.batch) + 1
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x1[rootindex[num_batch]]
        # x = th.cat((x, root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)    #(nodes_number,64)

#need change

        x = self.conv2(x, edge_index, edge_weight=edge_pred)

        x = F.relu(x)
    # rootindex feature pour out
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x, root_extend), 1)

        x_global = scatter_mean(x, data.batch, dim=0)    #[128,64]
        return x_global,x

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)    # [5617,1,64]
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)
        edge_pred = th.mean(edge_pred, dim=-1).squeeze(1)
        # w_mean = self.W_mean(x_ij)
        # w_bias = self.W_bias(x_ij)
        # b_mean = self.B_mean(x_ij)
        # b_bias = self.B_bias(x_ij)
        # logit_mean = w_mean * sim_val + b_mean
        # logit_var = th.abs(th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias)))
        #t= (sim_val**2)*th.exp(w_bias)+th.exp(b_bias)
        # edge_y = th.normal(logit_mean, logit_var)
        # edge_y = th.sigmoid(edge_y)
        # edge_y = self.fc2(edge_y)
        # logp_x = F.log_softmax(edge_pred, dim=-1)
        # p_y = F.softmax(edge_y, dim=-1)
        # edge_loss = self.eval_loss(logp_x, p_y)
        return edge_pred

def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    Eq = F.softplus(-q_samples) + q_samples

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_infer, l_enc_rand):

    '''
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]  #128
    num_nodes = l_enc.shape[0]   #numnodes

    pos_mask = th.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = th.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = th.mm(l_enc, g_enc.t())
    res_two = th.mm(l_enc_infer, g_enc.t())
    res_three = th.mm(l_enc_rand, g_enc.t())

    E_pos = get_positive_expectation((res + res_two + res_three) * pos_mask / 3, measure,
                                     average=False).sum()
    E_pos = E_pos / num_nodes

    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))     #one batch = 128, one is postive,others are negative

    return E_neg - E_pos

class EBGCN(th.nn.Module):
    def __init__(self, args):
        super(EBGCN, self).__init__()
        self.args = args
        self.TDrumorGCN = TDrumorGCN(args)
        self.fc = th.nn.Linear((args.hidden_features + args.output_features)*2, args.num_class)
        self.local_d = FF(args.hidden_features)
        self.global_d = FF(args.hidden_features)
        self.init_emb()
    def init_emb(self):  #?
        initrange = -1.5 / self.args.hidden_features
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        row_len = data.edge_index[0].shape
        rand_edge_weigth =th.rand(row_len).to(self.args.device)

        rand_x,rand_M = self.TDrumorGCN(data,edge_weight=rand_edge_weigth,edge_infer=False)
        infer_x,infer_M= self.TDrumorGCN(data,edge_weight=None,edge_infer=True)
        original_x,M = self.TDrumorGCN(data,edge_weight=None,edge_infer=False)

        g_enc = self.global_d(original_x)
        l_enc = self.local_d(M)
        # g_enc_infer = self.global_d(infer_x)  # feed forward
        l_enc_infer = self.local_d(infer_M)
        # g_enc_rand = self.global_d(rand_x)
        l_enc_rand = self.local_d(rand_M)

        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure, l_enc_infer, l_enc_rand)
        # self.x = th.cat(TD_x, 1)
        # out = self.fc(self.x)
        # out = F.log_softmax(out, dim=1)
        return local_global_loss
class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)