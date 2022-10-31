from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

#设置gpu设备
device = 3
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

path='/home/xzb/GCond/HW2/datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='/home/xzb/GCond/HW2/results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0].to(device)
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图
print(data.adj_t)

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']
result_dir = prepare_folder(dataset_name,'gcn')

from torch_geometric.nn import GCNConv,GATConv,SAGEConv,SGConv
from torch_sparse import SparseTensor
from typing import Union, Tuple
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i+2](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=1)

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
#         super().__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
#         self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.conv1 = GATConv(hidden_channels, hidden_channels, heads, dropout=dropout)
#         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
#                              concat=False, dropout=dropout)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         for i, lin in enumerate(self.lins):
#             x = lin(x)
#             x = self.bns[i](x)
#             x = F.elu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return torch.log_softmax(x, dim=-1)

#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         for lin in self.lins:
#             lin.reset_parameters()

# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(SAGE, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels))
#         self.convs.append(SAGEConv(hidden_channels, out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()

#     def forward(self, x, edge_index, edge_weight=None):
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index, edge_weight)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index, edge_weight)
#         return torch.log_softmax(x, dim=-1)

# class SGC(nn.Module):
    
#     def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
#         super(SGC, self).__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
#         self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SGConv(hidden_channels, out_channels, K, cached=False))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         for lin in self.lins:
#             lin.reset_parameters()

#     def forward(self, x, adj_t):
#         for i, lin in enumerate(self.lins):
#             x = lin(x)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x.log_softmax(dim=1)

# class MLP(torch.nn.Module):
#     def __init__(self
#                  , in_channels
#                  , hidden_channels
#                  , out_channels
#                  , num_layers
#                  , dropout
#                  , batchnorm=True):
#         super(MLP, self).__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
#         self.batchnorm = batchnorm
#         if self.batchnorm:
#             self.bns = torch.nn.ModuleList()
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         for _ in range(num_layers - 2):
#             self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
#             if self.batchnorm:
#                 self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         if self.batchnorm:
#             for bn in self.bns:
#                 bn.reset_parameters()

#     def forward(self, x):    
#         for i, lin in enumerate(self.lins[:-1]):
#             x = lin(x)
#             if self.batchnorm:
#                 x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lins[-1](x)
#         return F.log_softmax(x, dim=-1)

def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    
    # out = model(data.x[train_idx],data.adj_t[train_idx,train_idx])
    out = model(data.x[train_idx])

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            
            # out = model(data.x[node_id],data.adj_t[node_id,node_id])
            out = model(data.x[node_id])
            y_pred = out.exp()  # (N,num_classes)
            
            losses[key] = F.nll_loss(out, data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred)[eval_metric]

    return eval_results, losses, y_pred





#训练模型并且进行测试
args = {
    'lr': 0.01
    , 'num_layers': 3
    , 'hidden_channels':256
    , 'dropout': 0.5
    , 'heads': 4
    , 'weight_decay': 5e-7
                  }
epochs = 2000
log_steps =10 # log记录周期

model = GCN(in_channels=data.x.size(-1), hidden_channels=args['hidden_channels'], out_channels=nlabels, num_layers=args['num_layers'], dropout=args['dropout']).to(device)
# model = GAT(in_channels=data.x.size(-1), hidden_channels=args['hidden_channels'], out_channels=nlabels, heads=args['heads'], dropout=args['dropout']).to(device)
# model=SGC(in_channels=data.x.size(-1), hidden_channels=args['hidden_channels'], out_channels=nlabels, K=2, dropout=args['dropout']).to(device)
# model = MLP(in_channels=data.x.size(-1), hidden_channels=args['hidden_channels'], out_channels=nlabels, num_layers=args['num_layers'], dropout=args['dropout'], batchnorm=True).to(device)
eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)
print(sum(p.numel() for p in model.parameters()))  #模型总参数量

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
best_valid = 0
min_valid_loss = 1e8

i=13
model.load_state_dict(torch.load(save_dir+'/model'+str(i)+'.pt'))
for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir+'/model'+str(i)+'.pt') #将表现最好的模型保存

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
              f'Valid: {100 * valid_eval:.3f} ')

torch.cuda.empty_cache()
model.load_state_dict(torch.load(save_dir+'model'+str(i)+'.pt'))
out = model(data.x,data.adj_t)
output_numpy=out.detach().cpu().numpy()
np.save(save_dir+'out_numpy'+str(i)+'.npy',output_numpy,allow_pickle=True)
