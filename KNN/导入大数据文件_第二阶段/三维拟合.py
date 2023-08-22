import numpy
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

""" 
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 通过 KNN 来研究 父母的身高， 孩子的年龄，孩子的性别对孩子的身高的影响
        
"""


    # 前面是 年龄 ， 后面是 性别 为 男 的时候 表示为 1 ，性别为女的时候 表示为  0
def createData():
    age_man = np.array([[12, 1], [23, 0], [23, 1], [25, 1], [30, 0]])
    high = np.array([120, 169, 182, 188, 170])
    return age_man, high


if __name__ == '__main__':
    age_man, high = createData()
    age_man_test = input("请按这种格式[n,m]来输入年龄和性别：")
    age_man_test = eval(age_man_test)  # 将输入的字符串转换为列表

    k = int(input("请输入K值："))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(age_man, high)

    high_pred = knn.predict([age_man_test])
    print("预测的身高为:", high_pred[0])
