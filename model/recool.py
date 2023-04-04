'''
再冷盘管建模
input   盘管前空气温度
        盘管前空气含湿量
        空气质量流速
        调节阀开度
output  盘管供冷量 = 预处理后空气总焓 - 再冷后空气总焓
'''

import pandas as pd
import numpy
import seaborn as sns

data_path = './data/recool/2023_04_03.csv'

def recool():
    print("调用再冷模型---------------")
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.isnull().sum())
