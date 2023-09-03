import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

""" 
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 基于感知器构建的简单的分类器模型，我们来计算的是
    

"""



'''
    第一部分 ： 随机生成一下数据
    随机生成的一些数据，进行拟合，
'''
# 生成一些示例数据
# 从CSV文件加载数据，假设CSV文件名为data.csv
data = pd.read_excel('data.xlsx')


# 提取特征X（假设X包含在所有列中，除了最后一列）
X = data.iloc[:, :-1].values

# 提取标签y（假设标签y在最后一列）
y = data.iloc[:, -1].values


# 将数据转换为PyTorch张量
X = X.astype(np.float32)
  # 使用权重矩阵的数据类型来转换x的数据类型
X = torch.sigmoid(self.fc(X))

y = torch.tensor(y)

# 创建一个单层神经网络模型
class SingleLayerNN(nn.Module):
    def __init__(self):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(2, 1)  # 2个输入特征，1个输出

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

# 初始化模型
model = SingleLayerNN()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 尝试适度增大学习率

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y.view(-1, 1))  # 将标签视图重塑为列向量

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.float32)
    predicted = model(test_inputs)
    predicted = (predicted > 0.5).float()  # 大于0.5的值预测为1，否则预测为0
    print("Predictions:")
    print(predicted)
