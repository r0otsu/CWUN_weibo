# 训练、测试GCN模型
# -*- coding:utf-8 -*-
import time
import networkx as nx
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import dataset


# 可视化
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()
def visualize_embedding(h, color, save_path, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
        # plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    else:
        pass

    if save_path is not None:
        plt.savefig(save_path)
    else:
        pass
    plt.show()


# GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, num_classes)
        self.norm = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


# 加载数据函数
def load_data(name):
    # dataset = Planetoid(root='./' + name + '/', name=name)
    dataset = name
    # _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = dataset.to(_device)
    data = dataset.data
    num_node_features = dataset.data.num_node_features
    num_classes = dataset.num_classes
    return data, num_node_features, num_classes


# 训练
train_loss = []
test_acc = []
def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(200):
        out = model(data)
        optimizer.zero_grad()
        # loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        prob_out = F.softmax(out[data.train_mask], dim=1)
        labels = data.y[data.train_mask].long()
        loss = loss_function(prob_out, labels)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        acc = test(model, data)
        test_acc.append(acc)
        # print(test_acc)
        print('Epoch:{:03d}; loss:{:.4f}; testAcc:{:.4f}'.format(epoch, loss.item(), acc))

        with open("./model_result/train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss))
        with open("./model_result/test_acc.txt", 'w') as test_ac:
            test_ac.write(str(test_acc))

        if epoch % 10 == 0:
            visualize_embedding(out, color=data.y, save_path=None, epoch=epoch, loss=loss)
            time.sleep(0.3)
        torch.save(model.state_dict(), './model_result/gcn_model.pth')
    # return loss,


# 测试
def test(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc
    # print('GCN Accuracy: {:.4f}'.format(acc))



def main(dataset):
    data, num_node_features, num_classes = load_data(dataset)
    print(data, num_node_features, num_classes)
    _device = 'cpu'
    device = torch.device(_device)
    model = GCN(num_node_features, num_classes).to(device)
    train(model, data, device)
    test(model, data)


# 加载自己创建的pyg数据库
dataset = dataset
if __name__ == '__main__':
    main(dataset)
