# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
import time

"""
	1,author : S_zhi 
	2,E-mail : feng978744573@163.com 
	3,project : KNN 图像处理第一阶段 ， 数字识别 0 — 9 移植项目（学习）
		功能 ： 对数字进行识别 , 误差分析 , 时长
		名词解释 ：
		>train 文件 训练数据  
		>test 文件 测试数据
"""



"""
本函数为KNN 的决策依据，移植函数
————————————————————————————————————————

我们称之为投票函数

 ————————————————————————————————————————
"""
def classify0(inX, dataSet, labels, k):
	#numpy函数shape[0]返回dataSet的行数
	dataSetSize = dataSet.shape[0]
	#在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	#二维特征相减后平方
	sqDiffMat = diffMat**2
	#sum()所有元素相加,sum(0)列相加,sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	#开方,计算出距离
	distances = sqDistances**0.5
	#返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

"""
本函数为KNN 的决策依据的构建，移植函数
————————————————————————————————————————

函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量

 ————————————————————————————————————————
"""
def img2vector(filename):
	#创建1x1024零向量
	returnVect = np.zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#按行读取
	for i in range(32):
		#读一行数据
		lineStr = fr.readline()
		#每一行的前32个元素依次添加到returnVect中
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回转换后的1x1024向量
	return returnVect

"""
	handwritingClassTest 为主函数 完成程序的主要内容
 	1. 第一步 ： 提取文件的类型，数量，对文件进行每个分分布处理
  	2. 第二步 ： 完成训练集的构建
   	3. 第三步 ： 进入预测函数
"""
def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('train_data')
	m = len(trainingFileList)

	trainingMat = np.zeros((m, 1024))
	# 第一步
	for i in range(m):
		
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		# 存储每个训练集文件所属于的数字
		hwLabels.append(classNumber)
		trainingMat[i,:] = img2vector('train_data/%s' % (fileNameStr))
		# 完成训练集的构建
	
	testFileList = listdir('test_data')
	errorCount = 0.0
	mTest = len(testFileList)

	for i in range(mTest):
		
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		
		vectorUnderTest = img2vector('test_data/%s' % (fileNameStr))
		# 进入预测函数 
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))


if __name__ == '__main__':
	start_time = time.time()

	handwritingClassTest()
	# 主函数 ： 对数字数据进行训练 , 对数字数据集进行测试 , 返回（每组的正确值 ,预测值  / 错误的正确值和预测值） , 返回误差值。

	end_time = time.time()
	execution_time = end_time - start_time
	print(f"执行时间为: {execution_time} 秒")
	# 返回时间
