import pandas as pd
import numpy as np



# 从CSV文件加载数据，假设CSV文件名为data.csv
data = pd.read_excel('data.xlsx')# 或者 encoding='latin1'


# 提取特征X（假设X包含在所有列中，除了最后一列）
X = data.iloc[:, :-1].values

# 提取标签y（假设标签y在最后一列）
y = data.iloc[:, -1].values

# 打印特征和标签
print("特征 X:")
print(X)

print("标签 y:")
print(y)
