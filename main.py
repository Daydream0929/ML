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

# 处理excel数据中的错误
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# 数据处理
def data_process():
    data = pd.read_excel(data_source_path).to_numpy()
    X = data[:, [1, 2, 3, 6, 7, 8, 9, 14, 19, 21, 22]]
    y = data[:, [12]] + data[:, [13]] + data[:, [18]] + data[:, [20]] 
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    

# 模型训练
def model_train():
    for md in model:
        if md == "损失函数":
            alpha = 0.1
            lasso = linear_model.Lasso(alpha=alpha)
            # regr = lasso.fit(X_train, y_train)
            #model_predict(regr)
        
        
# 模型预测
def model_predict(regr):
    y_pred = regr.predict(X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
def main():
   data_process()
   model_train()
    #model_predict()

if __name__ == "__main__":
    main()