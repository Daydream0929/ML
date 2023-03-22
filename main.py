import pandas as pd

data_path = './data/data1.xlsx'  # 存储数据路径
  
# 数据处理
def data_process():
    data = pd.read_excel(data_path)
    print(data)

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