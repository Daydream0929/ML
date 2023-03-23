import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

test_path = './data/data0.xlsx'  # 用来测试的数据
data_source_path = './data/data.xlsx'  # 存储数据路径

X_train = []  # 特征向量训练集
X_test = []   # 特征向量测试集
y_train = []  # 预测变量训练集
y_test = []   # 预测变量测试集
model = ["线性回归", "损失函数"]    # 模型选取

# 数据处理
def data_process():
    # 从excel中读取数据并转化为numpy
    data = pd.read_excel(data_source_path).to_numpy()

    # 从源数据中提取特征向量和目标向量(str类型)
    X_str = data[:, [1, 2, 3, 6, 7, 8, 9, 14, 19, 21, 22]]
    y_str = data[:, [12]] + data[:, [13]] + data[:, [18]] + data[:, [20]] 

    # 将特征向量和目标向量str转化为float
    X = np.zeros((X_str.shape[0], X_str.shape[1]), dtype=float)
    y = np.zeros((y_str.shape[0], y_str.shape[1]), dtype=float)  
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            try:
               X[i][j] = float(X_str[i][j])  
            except ValueError:
               X[i][j] = 0.0
    for i in range(0, y.shape[0]):
        for j in range(0, y.shape[1]):
            try:
                y[i][j] = float(y_str[i][j])  
            except ValueError:
                y[i][j] = 0.0

    # 将源数据集划分为训练集和测试集
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

# 模型训练
def model_train():
    for md in model:
        if md == "损失函数":
            print("采用Lasso模型: ")
            alpha = 0.1
            lasso = linear_model.Lasso(alpha=alpha)
            regr = lasso.fit(X_train, y_train)
            model_predict(regr)
        elif md == "线性回归":
            print("采用线性回归模型: ")
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            model_predict(regr)
        elif md == "线性回归":
            print("采用线性回归模型: ")
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            model_predict(regr)
        elif md == "线性回归":
            print("采用线性回归模型: ")
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            model_predict(regr)
        elif md == "线性回归":
            print("采用线性回归模型: ")
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            model_predict(regr)
        elif md == "线性回归":
            print("采用线性回归模型: ")
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            model_predict(regr)        
        
        
# 模型预测
def model_predict(regr):
    y_pred = regr.predict(X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    # plt.scatter(X_test, y_test, color="black")
    # plt.plot(X_test, y_pred, color="blue", linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()
def main():
    data_process()
    model_train()
    # model_predict()

if __name__ == "__main__":
    main()