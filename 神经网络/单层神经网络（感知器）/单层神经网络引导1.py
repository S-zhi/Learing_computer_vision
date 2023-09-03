import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成一些示例数据
# 假设我们的任务是将输入数据分为两个类别（0和1）
# 这里生成了一些示例数据
np.random.seed(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# 将数据转换为PyTorch张量
X = torch.tensor(X)
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
optimizer = optim.SGD(model.parameters(), lr=0.1)

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
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predicted = model(test_inputs)
    predicted = (predicted > 0.5).float()  # 大于0.5的值预测为1，否则预测为0
    print("Predictions:")
    print(predicted)
