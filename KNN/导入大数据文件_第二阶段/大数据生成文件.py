import pandas as pd
import random

# 创建空的数据字典
data = {
    '父母平均身高': [],
    '孩子年龄': [],
    '孩子性别': [],
    '孩子身高': []
}

# 生成100组数据
for _ in range(100):
    # 随机生成父母平均身高在150到190之间的整数
    parent_height = random.randint(150, 190)

    # 随机生成孩子年龄在1到18之间的整数
    child_age = random.randint(1, 18)

    # 随机生成孩子性别，1表示男孩，0表示女孩
    child_gender = random.choice([0, 1])

    # 根据父母平均身高、孩子年龄和性别生成孩子身高的简单模拟（仅用于示例，实际情况可能更复杂）
    child_height = parent_height - 10 + (child_age / 2) + (child_gender * 5)

    # 将生成的数据添加到数据字典中
    data['父母平均身高'].append(parent_height)
    data['孩子年龄'].append(child_age)
    data['孩子性别'].append(child_gender)
    data['孩子身高'].append(child_height)

# 创建数据框
df = pd.DataFrame(data)

# 打印前几行数据
print(df)
df.to_csv('sample_data.csv', index=False)
