# 将特征使用snownlp及bert进行编码
import os
import numpy as np
import pandas as pd
import torch
from snownlp import SnowNLP
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def to_node_features(csv_file, data_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据
    comments = pd.read_csv(csv_file, encoding='utf-8')['comment'].values
    times = pd.read_csv(csv_file, encoding='utf-8')['created_at'].values
    model_name = "./bert-model"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    # 确定输出文件目录
    output_dir = './weibo_data' if data_type == 'train' else './predict_data'
    os.makedirs(output_dir, exist_ok=True)

    # 对comment的编码
    with open(f"{output_dir}/comment_embedding.txt", "w", encoding='utf-8') as file:
        for comment in tqdm(comments, desc="Processing comments"):
            # Tokenize the comment
            batch_tokenized = tokenizer.batch_encode_plus([comment], add_special_tokens=True,
                                                          max_length=148, padding='max_length',
                                                          truncation='longest_first')
            input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
            hidden_outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state.to(
                'cpu').detach().numpy()
            last_hidden_state = hidden_outputs[:, -1, :]
            np.savetxt(file, last_hidden_state, delimiter=',')
    # 读取每句embedding
    with open(f"{output_dir}/comment_embedding.txt", "r", encoding='utf-8') as file:
        for line in file:
            embedding = [float(value) for value in line.strip().split(',')]
            embeddings = np.array(embedding).reshape(1, -1)
            # print(embeddings)

    # 标准化
    def standardization(data):
        mean = np.mean(data)
        std = np.std(data)
        standardized_data = [(x - mean) / std for x in data]
        return standardized_data

    times = standardization(times)
    np.savetxt(f"{output_dir}/time_embedding.txt", times, fmt='%f')
    time_embedding = torch.tensor(times, dtype=torch.float)
    time_feature = torch.unsqueeze(time_embedding, dim=1)

    # 对time的编码
    # with open(f"{output_dir}/time_embedding.txt", "w", encoding='utf-8') as file:
        # for time in times:
        #     # Tokenize the comment
        #     batch_tokenized = tokenizer.batch_encode_plus([time], add_special_tokens=True,
        #                                                   max_length=148, padding='max_length',
        #                                                   truncation='longest_first')
        #     input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        #     attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        #     hidden_outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state.to(
        #         'cpu').detach().numpy()
        #     last_hidden_state = hidden_outputs[:, -1, :]
        #     np.savetxt(file, last_hidden_state, delimiter=',')
    # # 读取每句embedding
    # with open(f"{output_dir}/time_embedding.txt", "r", encoding='utf-8') as file:
    #     for line in file:
    #         embedding = [float(value) for value in line.strip().split(',')]
    #         embeddings = np.array(embedding).reshape(1, -1)
    #         # print(embeddings)

    # 将comment_embedding转为tensor
    comment_embedding = []
    # 打开文件并逐行读取数据
    with open(f"{output_dir}/comment_embedding.txt", "r", encoding="utf-8") as f:
        for line in f:
            # 首先按空格拆分每行
            embeddings = []
            for part in line.split():
                # 对每个子字符串按逗号拆分，并转换为浮点数
                embeddings.extend([float(x.strip()) for x in part.split(',')])
            comment_embedding.append(embeddings)
    # 将列表转换为 PyTorch 张量
    comment_feature = torch.tensor(comment_embedding, dtype=torch.float)
    # print(comment_embedding)


    # # 将time_embedding转为tensor
    # time_embedding = []
    # # 打开文件并逐行读取数据
    # with open(f"{output_dir}/time_embedding.txt", "r", encoding="utf-8") as f:
    #     for line in f:
    #         # 首先按空格拆分每行
    #         embeddings = []
    #         for part in line.split():
    #             # 对每个子字符串按逗号拆分，并转换为浮点数
    #             embeddings.extend([float(x.strip()) for x in part.split(',')])
    #         time_embedding.append(embeddings)

    # # 将列表转换为 PyTorch 张量
    # time_feature = torch.tensor(time_embedding, dtype=torch.float)
    # print(time_embedding)
    # print(comment_embedding.shape)
    # print(time_embedding.shape)
    s = []
    for c in comments:
        score = SnowNLP(c).sentiments
        s.append(score)

    sentiment_scores = s
    # data.to_excel("data_result.xlsx", index=False)  # 导出标记情感标签的评论数据
    # 将sentiment_score单独转存
    # sentiment_embedding = pd.read_csv(csv_file, encoding='utf-8')['sentiment_score'].values
    np.savetxt(f"{output_dir}/sentiment_embedding.txt", sentiment_scores, fmt='%f')
    sentiment_embedding = torch.tensor(sentiment_scores, dtype=torch.float)
    sentiment_feature = torch.unsqueeze(sentiment_embedding, dim=1)
    node_features = torch.cat((sentiment_feature, time_feature, comment_feature), -1)

    return node_features


# node_features = to_node_features(csv_file="label.csv", data_type="train")
