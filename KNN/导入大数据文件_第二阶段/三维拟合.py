""" 
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 通过 KNN 来研究 父母的身高， 孩子的年龄，孩子的性别对孩子的身高的影响
    4.名词解释 : train.csv 训练数据（在大数据生成文件.py中自动生成 ，或者选用数据集中的数据，改名称，本程序不自带）
                test.csv 尝试的数据（大数据生成文件.py中自动生成）
                result.csv 生成的最终的文件 
    
"""
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


loaded_data = pd.read_csv('train.csv')

父母平均身高 = loaded_data['父母平均身高'].to_numpy()
孩子年龄 = loaded_data['孩子年龄'].to_numpy()
孩子性别 = loaded_data['孩子性别'].to_numpy()
孩子身高 = loaded_data['孩子身高'].to_numpy()

# 创建一个DataFrame来存储相关的内容  导入 pandas文件才能使用 
data = pd.DataFrame({
'父母平均身高': 父母平均身高,
'孩子年龄': 孩子年龄,
'孩子性别': 孩子性别,
'孩子身高': 孩子身高
})

# 第一步：初始化KNN回归模型
knn = KNeighborsRegressor(n_neighbors=3)  

# 第二步：使用所有的数据来训练KNN回归模型
X = data[['父母平均身高', '孩子年龄', '孩子性别']].values
y = data['孩子身高'].values
knn.fit(X, y)

test_data = pd.read_csv('test.csv')
X_test = test_data[['父母平均身高', '孩子年龄', '孩子性别']].values

# 第三步：进行预测
predicted_y = knn.predict(X_test)
test_data['预测身高'] = predicted_y

# 保存测试结果为CSV文件
test_data.to_csv('results.csv', index=False)
