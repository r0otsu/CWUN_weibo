import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import torch_geometric.transforms as T
# from embedding import time_embedding, comment_embedding, to_node_features
from embedding import to_node_features
from matrix import to_edge_index


class WeiboDataset(InMemoryDataset):
    def __str__(self):
        return f'WeiboDataset({len(self)}, root={self.root})'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     pass

    def process(self):
        data_list = []
        data = pd.read_csv("train_.csv")
        node_features = to_node_features(csv_file="train_.csv", data_type="train")
        # sentiment_scores = data['sentiment_score']
        # labels = data['label']
        # 特征拼接
        # sentiment_feature = torch.tensor(sentiment_scores, dtype=torch.float)
        # sentiment_feature = torch.unsqueeze(sentiment_feature, dim=1)
        # time_feature = torch.tensor(time_embedding, dtype=torch.float)
        # comment_feature = torch.tensor(comment_embedding, dtype=torch.float)
        # node_features = torch.cat((sentiment_feature, time_feature, comment_feature), -1)

        # 节点之间的边
        # edge_index = torch.tensor(Gt, dtype=torch.long)
        Gt, Gc, Gs = to_edge_index(csv_file="train_.csv")
        edge_index = torch.tensor(Gt, dtype=torch.long)
        # 标签
        # node_labels = torch.tensor(labels, dtype=torch.float)
        # 获得data数据
        # data = Data(x=node_features, edge_index=edge_index, y=node_labels)
        data = Data(x=node_features, edge_index=edge_index)
        #
        print(data)
        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        node_data = self.data_list[idx]
        x = node_data['x']
        y = node_data['y']
        data = Data(x=x, y=y)
        return data
#
#
dataset = WeiboDataset('Mydata')
#
# 将数据集分为0.9的训练集和0.1的测试集
# data = T.RandomNodeSplit(num_test=0.1, num_val=0)(dataset.data)
# print(data)
