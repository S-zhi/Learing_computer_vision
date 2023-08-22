import pandas as pd
import random

# 创建空的数据字典
data = {
    '父母平均身高': [],
    '孩子年龄': [],
    '孩子性别': [],
    '孩子身高': []
}

"""
    
"""
#随机生成函数 
for _ in range(100):
    # 随机生成父母平均身高在150到190之间的整数
    parent_height = random.randint(150, 190)

    # 随机生成孩子年龄在1到18之间的整数
    child_age = random.randint(1, 18)

    # 随机生成孩子性别，1表示男孩，0表示女孩
    child_gender = random.choice([0, 1])

   
    if child_age <= 18:
        parent_height -= 50
        child_height = 40 + parent_height / 16 * child_age + (child_gender * 5)
    else:
        parent_height -= 50
        child_height = 50 + parent_height / 20 * 21 + (child_gender * 5)
    parent_height += 50
    """
    拟合数据 ： 首先是使用分支结构来设置的数据，其实本测试数据不用else : 之后的语句
    但为了保证数据的合理性，本数据扩展了孩子 18岁之后的身高情况.
    
    """


    
    # 将生成的数据添加到数据字典中 相当于 map 字典
    data['父母平均身高'].append(parent_height)
    data['孩子年龄'].append(child_age)
    data['孩子性别'].append(child_gender)
    data['孩子身高'].append(child_height)

# 创建数据框
df = pd.DataFrame(data)

# 打印前几行数据
print(df)
df.to_csv('sample_data.csv', index=False)
