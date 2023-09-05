import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

"""
    
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 基于感知器构建的神经网络模型，由于本数据集合比较差劲，所以当前仅仅当作学习神经网络的方法之一。
    我们只要使用这个项目来学习神经网络的基本的内容就好
    主要分成下面的几步
    1，创建模型
    2，初始化模型（简单的初始化 ， 一些常规函数的定义 ）
    3，训练模型 
    4，测试模型

"""


# 完成数据的导入
data = pd.read_excel('data.xlsx')

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values

X = X.astype(np.float32)
y = torch.tensor(y, dtype=torch.float32)
# 完成数据的导入


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
optimizer = optim.SGD(model.parameters(), lr=0.9)  # 尝试适度增大学习率
# 初始化模型



# 训练模型
num_epochs = 10000000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(torch.tensor(X, dtype=torch.float32))
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
