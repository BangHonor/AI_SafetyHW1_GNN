DGraph-Fin 是一个由数百万个节点和边组成的有向无边权的动态图。它代表了Finvolution Group用户之间的社交网络，其中一个节点对应一个Finvolution 用户，从一个用户到另一个用户的边表示该用户将另一个用户视为紧急联系人。 下面是位于dataset/DGraphFin目录的DGraphFin数据集的描述: x: 20维节点特征向量 y: 节点对应标签，一共包含四类。其中类1代表欺诈用户而类0代表正常用户(实验中需要进行预测的两类标签)，类2和类3则是背景用户，即无需预测其标签。 edge_index: 图数据边集,每条边的形式(id_a,id_b)，其中ids是x中的索引 edge_type: 共11种类型的边 edge_timestamp: 脱敏后的时间戳 train_mask, valid_mask, test_mask: 训练集，验证集和测试集掩码

该数据集要实现的目标是，通过深度学习算法将正常用户及欺诈用户进行划分，解决的思路如下： 1.图数据优先选用GNN，该案例图数据的邻接矩阵规模为size=(3700550, 3700550), nnz=7994520, density=0.00%，边十分的稀疏，越稀疏的图意味着GNN很容易会出现过平滑现象（节点的聚合半径变小，一旦达到某个阈值，节点的接受域将覆盖全图节点，此時所有的节点都趋于相等），所以不能用太深的GNN，暂定使用GCN，层数为2/3层。 2.用MLP在X上可以提取比较好的feature，该feature可以作为GCN的输入。 3.结合GNN与MLP两者的优势，我们先用两层的MLP对feature进行特征提取，然后再使用GCN进行图上的卷积运算。

参数选择如下：
args = {
    'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels':128
    , 'dropout': 0.5
    , 'weight_decay': 5e-7
                  }
参数量为：两层线性层≈20*256+256*256，两层卷积≈256*256*2，总的参数量约为200,000。

优势：该算法的优势在于能够很好地提取feature，并利用GCN聚合local structure以及feature信息，得到每一个节点的embedding，能够简单高效地对节点进行分类任务。

不足：使用full-batch进行训练，可能会消耗比较大的内存；将所有的边都视为一样的，但是实际上不同的边应该有不同的权重，但是GAT将会占用更多的内存，在GPU允许的情况下应该优先GAT；GCN聚合邻接节点信息的方式是通过加权平均，聚合方式较为简单。
