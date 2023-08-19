import numpy
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np
"""
    1.author : S_zhi   
    2.E-mail : feng978744573@163.com
    3.project :  通过 KNN 来研究 人的年龄对人的身高的影响 
    4.op : 训练值 ---> createData 年龄 | 身高  
           训练方法 --->  KNN库中的函数  
"""
def createData() :
     age = np.array([[1], [4], [7], [9], [10], [12], [15], [18], [20], [24], [30], [40], [50]])
     high = np.array([30, 50, 80, 90, 100,120, 140, 180, 185, 178, 188, 174, 168])
     return age , high

if __name__ == '__main__' :
    age , high = createData()
    # 再次赋值 
    age_test = int (input("请输入一个想要预测的年龄值"))
    k = int (input("输入一个想要预测的K 值"))
    knn = KNeighborsClassifier(n_neighbors= k ) 
    # 读入的 K 值 
    knn.fit(age, high)  
    # 进行模拟模型
    high_pred = knn.predict(np.array([[age_test]]))
    
    print(high_pred)
