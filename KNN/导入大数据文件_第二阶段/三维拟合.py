""" 
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 通过 KNN 来研究 父母的身高， 孩子的年龄，孩子的性别对孩子的身高的影响
        
"""


import numpy
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# 创建一个DataFrame 来存储相关的内容   
data = pd.DataFrame({
    '父母平均身高': parent_height,
    '孩子年龄': child_age,
    '孩子性别': child_gender,
    '孩子身高': child_height
})
    # 前面是 年龄 ， 后面是 性别 为 男 的时候 表示为 1 ，性别为女的时候 表示为  0
def createData():
    loaded_data = pd.read_csv('family_data.csv')
    父母平均身高 = loaded_data['父母平均身高'].to_numpy()
    孩子年龄 = loaded_data['孩子年龄'].to_numpy()
    孩子性别 = loaded_data['孩子性别'].to_numpy()
    孩子身高 = loaded_data['孩子身高'].to_numpy()
    return 父母平均身高,孩子年龄,孩子性别,孩子身高


if __name__ == '__main__':
    parent_high , child_age , child_is_man_or_women , child_high = createData()

    #导入创建的数据集合 
    
