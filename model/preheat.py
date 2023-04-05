'''
预热盘管建模
input   盘管前空气温度
        盘管前空气含湿量
        空气质量流速
        调节阀开度
output  盘管供热量 = 预处理空气总焓 - 外气总焓
'''
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from algorithms import ANN_Model
from torch import nn

data_path = './data/preheat/2023_04_03.csv'

def preheat():
    # 提示调用预热模型
    print("调用预热模型---------------")

    # 读取原始数据
    data = pd.read_csv(data_path, skiprows=lambda x: x > 4000)

    # 提取输入 输出
    X, y = data.drop('Q', axis=1).values, data['Q'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 将数据转化为张量格式
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test  = torch.FloatTensor(y_test)

    # 定义模型
    torch.manual_seed(20)
    model = ANN_Model()

    # Backward Propagation - Define the Loss Function && Optimizer
    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练
    epochs = 400
    final_losses = []
    for i in range(epochs):
        i = i + 1
        y_pred = model.forward(X_train)
        loss = loss_function(y_pred, y_train)
        final_losses.append(loss.item())
        if i % 10 == 1:
            print('Epoch number: {} and the loss : {}'.format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), final_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
