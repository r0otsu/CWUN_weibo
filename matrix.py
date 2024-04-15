# 获得pyg可用的节点稠密tensor矩阵
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm
import torch


# 定义用于内容、情感和时间级别的函数
def tf_similarity(review_i, review_j):
    def add_space(s):
        return ' '.join(list(s))

    review_i, review_j = add_space(review_i), add_space(review_j)  # 在字中间加上空格
    cv = CountVectorizer(tokenizer=lambda s: s.split())  # 转化为TF矩阵
    corpus = [review_i, review_j]
    vectors = cv.fit_transform(corpus).toarray()  # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def to_edge_index(csv_file):
    data = pd.read_csv(csv_file)
    # 初始化变量和参数
    reviews = data['comment']  # 评论列表
    # sentiment_scores = sentiment_scores
    create_times = data['created_at']

    sentiment_scores = []
    # 打开txt文件
    with open('./weibo_data/sentiment_embedding.txt', 'r') as file:
        for line in file:
            sentiment_scores.append(float(line.strip()))
    print(sentiment_scores)


    # print(create_times)
    sigma_c = 0.5  # 内容相似度的阈值
    sigma_s = 0.2  # 情感偏差的阈值
    sigma_t = 50  # 时间偏差的阈值

    # 构建弱关系图
    Gc = np.zeros((len(reviews), len(reviews)))  # 内容级别图
    Gs = np.zeros((len(reviews), len(reviews)))  # 情感级别图
    Gt = np.zeros((len(reviews), len(reviews)))  # 时间级别图

    # 获得邻接矩阵
    for i in range(len(reviews)):
        for j in range(len(reviews)):
            if i != j:
                # 内容级别图
                if tf_similarity(reviews[i], reviews[j]) > sigma_c:
                    Gc[i, j] = 1
                # 情感级别图
                if abs(sentiment_scores[i] - sentiment_scores[j]) < sigma_s:
                    Gs[i, j] = 1
                # 时间级别图
                if abs(create_times[i] - create_times[j]) < sigma_t:
                    Gt[i, j] = 1
        print(i)
    # print(Gt, Gc, Gs)

    # 转为tensor稠密矩阵
    Gt = coo_matrix(Gt)
    values_t = Gt.data
    indices = np.vstack((Gt.row, Gt.col))
    Gt = torch.LongTensor(indices)
    print(Gt)

    Gs = coo_matrix(Gs)
    values_s = Gs.data
    indices = np.vstack((Gs.row, Gs.col))
    Gs = torch.LongTensor(indices)
    # print(Gs)

    Gc = coo_matrix(Gc)
    values_c = Gc.data
    indices = np.vstack((Gc.row, Gc.col))
    Gc = torch.LongTensor(indices)
    # print(Gc)

    return Gt, Gc, Gs
