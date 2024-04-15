import time
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import nn
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import ClusterGCNConv
from dataset import dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


torch.manual_seed(16)
# 加载自己创建的pyg数据库
dataset = dataset
data = T.RandomNodeSplit(num_test=0.1, num_val=0)(dataset.data)
cluster_data = ClusterData(data, num_parts=128)
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
total_num_nodes = 0

for step, sub_data in enumerate(train_loader):
    print(f'step{step}:')
    print(f'current batch:{sub_data.num_nodes}')
    print(sub_data)
    total_num_nodes += sub_data.num_nodes
print(f'interated over {total_num_nodes} of {data.num_nodes} ')


def compute_davies_bouldin_index(embeddings, cluster_labels):
    db_index = davies_bouldin_score(embeddings, cluster_labels)
    return db_index


def compute_calinski_harabasz_index(embeddings, cluster_labels):
    ch_index = calinski_harabasz_score(embeddings, cluster_labels)
    return ch_index


def visualize_embedding(h, cluster_labels, save_path=None):
    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7, 7))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab10')
    plt.xticks([])
    plt.yticks([])
    # if save_path is not None:
    #     plt.savefig(save_path)
    plt.show()


class ClusterGCN(torch.nn.Module):
    def __init__(self, out_channels=770):
        super(ClusterGCN, self).__init__()

        self.conv1 = ClusterGCNConv(in_channels=770, out_channels=32, diag_lambda=1.0)
        self.conv2 = ClusterGCNConv(in_channels=32, out_channels=out_channels, diag_lambda=1.0)

        self.dp = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


model = ClusterGCN(out_channels=770)
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(reduction='mean')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


def train():
    model.train()
    loss_epoch = []
    silhouette_scores = []
    embeddings = []

    for sub_data in train_loader:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sub_data = sub_data.to(device)
        optimizer.zero_grad()
        out = model(sub_data.x, sub_data.edge_index)
        reconstruction_loss = F.mse_loss(out, sub_data.x)
        reconstruction_loss.backward()
        optimizer.step()
        loss_epoch.append(round(float(reconstruction_loss.cpu()), 4))

        # 聚类嵌入向量并计算轮廓系数
        embeddings.append(out.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    for n_clusters in range(2, 11):  # 尝试不同数量的聚类
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=50)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette))

    best_cluster = select_cluster_number(silhouette_scores)
    kmeans = KMeans(n_clusters=best_cluster)
    best_cluster_labels = kmeans.fit_predict(embeddings)

    return loss_epoch, best_cluster_labels, silhouette_scores, embeddings


# 选择轮廓系数最高的聚类数量
def select_cluster_number(silhouette_scores):
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return best_n_clusters

start_time = time.time()

for epoch in range(10):
    loss, best_cluster_labels, silhouette_scores, embeddings = train()
    loss_str = ', '.join([f'{l:.4f}' for l in loss])
    best_n_clusters, best_silhouette_score = max(silhouette_scores, key=lambda x: x[1])
    db_index = compute_davies_bouldin_index(embeddings, best_cluster_labels)
    ch_index = compute_calinski_harabasz_index(embeddings, best_cluster_labels)
    print(f"================Epoch: {epoch}================")
    print(f"Davies-Bouldin Index:{db_index:.4f}")
    print(f"Calinski-Harabasz Index:{ch_index:.4f}")
    print(f"best_n_clusters: {best_n_clusters}, best_silhouette_score: {best_silhouette_score:.4f}")
    print(f"Loss: {loss_str}")
    visualize_embedding(embeddings, best_cluster_labels, save_path=f'./embedding_visualization_epoch_{epoch}.png')


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
