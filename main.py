import pandas as pd
from sklearn.model_selection import train_test_split

test_path = './data/data0.xlsx'  # 用来测试的数据
data_source_path = './data/data.xlsx'  # 存储数据路径

# 数据处理
def data_process():
    data = pd.read_excel(data_source_path).to_numpy()
    X = data[:, [1, 2, 3, 6, 7, 8, 9, 14, 19, 21, 22]]
    y = data[:, [12]] + data[:, [13]] + data[:, [18]] + data[:, [20]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

# 模型训练
def model_train():
    pass

# 模型预测
def model_predict():
    pass

def main():
   data_process()
   model_train()
   model_predict()

if __name__ == "__main__":
    main()