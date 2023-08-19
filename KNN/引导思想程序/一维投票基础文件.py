import numpy
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np
"""
    author : S_zhi   
    E-mail : feng978744573@163.com
    project : age --->  high KNN  
    option : 数据集合的大小，相当于数据库中的测试数据的多少 （20） 
    进行猜测的值 :  N  >> 第一个输入的值
    K 值是可以流动的  >>  输入的第二个的值 。  
"""
def createData() :
     age = np.array([[1], [4], [7], [9], [10], [12], [15], [18], [20], [24], [30], [40], [50]])
     high = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
     return age , high

if __name__ == '__main__' :
    age , high = createData()
    age_test = int (input("请输入一个想要预测的年龄值"))
    k = int (input("输入一个想要预测的K 值"))
    knn = KNeighborsClassifier(n_neighbors= k )  # 这里设置K=3，你可以根据需要调整K的值
    knn.fit(age, high)  # X_train是特征数据，y_train是标签数据

    high_pred = knn.predict(np.array([[age_test]]))

    print(high_pred)
