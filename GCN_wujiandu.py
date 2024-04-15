import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from torch_geometric.nn import GCNConv
from dataset import dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_davies_bouldin_index(embeddings, cluster_labels):
    db_index = davies_bouldin_score(embeddings, cluster_labels)
    return db_index


def compute_calinski_harabasz_index(embeddings, cluster_labels):
    ch_index = calinski_harabasz_score(embeddings, cluster_labels)
    return ch_index


# 选择轮廓系数最高的聚类数量
def select_cluster_number(silhouette_scores):
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return best_n_clusters


def visualize_embedding(embeddings, cluster_labels, save_path=None):
    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7, 7))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab10')
    plt.xticks([])
    plt.yticks([])
    # if save_path is not None:
    #     plt.savefig(save_path)
    plt.show()


def load_data(name):
    # dataset = Planetoid(root='./' + name + '/', name=name)
    dataset = name
    # _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = dataset.to(_device)
    data = dataset.data
    num_node_features = dataset.data.num_node_features
    num_classes = dataset.num_classes
    return data, num_node_features, num_classes


# GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 训练
train_loss = []


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_function = torch.nn.MSELoss().to(device)  # 无监督学习的损失函数
    model.train()
    silhouette_scores = []
    embeddings_list = []
    for epoch in range(11):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out, data.x)  # 重建输入的邻接矩阵作为损失
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        embeddings_list.append(out.detach().cpu().numpy())
        # embeddings = np.concatenate(embeddings, axis=0)
        # if epoch % 10 == 0 or epoch == 199:
        embeddings = np.concatenate(embeddings_list, axis=0)
        for n_clusters in range(2, 11):  # 尝试不同数量的聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=50)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append((n_clusters, silhouette))

        best_cluster = select_cluster_number(silhouette_scores)
        kmeans = KMeans(n_clusters=best_cluster)
        best_cluster_labels = kmeans.fit_predict(embeddings)
        best_n_clusters, best_silhouette_score = max(silhouette_scores, key=lambda x: x[1])
        db_index = compute_davies_bouldin_index(embeddings, best_cluster_labels)
        ch_index = compute_calinski_harabasz_index(embeddings, best_cluster_labels)
        print(f"================Epoch: {epoch}================")
        print(f"Davies-Bouldin Index:{db_index:.4f}")
        print(f"Calinski-Harabasz Index:{ch_index:.4f}")
        print(f"best_n_clusters: {best_n_clusters}, best_silhouette_score: {best_silhouette_score:.4f}")
        print('Epoch:{:03d}; loss:{:.4f}'.format(epoch, loss.item()))
        # if epoch % 10 == 0:
        # 在 train 函数中调用 visualize_embedding 方法：
        visualize_embedding(embeddings, best_cluster_labels, save_path=f'./embedding_visualization_epoch_{epoch}.png')
        # 这里可以添加其他评估指标，如图的重构误差等


def main(dataset):
    data, num_node_features, _ = load_data(dataset)  # 不再需要 num_classes
    print(data, num_node_features)
    _device = 'cpu'
    device = torch.device(_device)
    model = GCN(num_node_features).to(device)
    train(model, data, device)

# 加载自己创建的pyg数据库
dataset = dataset
if __name__ == '__main__':
    main(dataset)