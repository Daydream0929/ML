'''
再热盘管建模
input   盘管前空气温度
        盘管前空气含湿量
        空气质量流速
        调节阀开度
output  盘管供热量 = 送风总焓 - 再冷后空气总焓
'''

import pandas as pd
import numpy
import seaborn as sns

data_path = './data/reheat/2023_04_03.csv'

def reheat():
    print("调用预冷模型---------------")
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.isnull().sum())
