# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # 引入归一化的包


def linearRegression():
	print(u"加载数据...\n")
	dataX = loadtxtAndcsv_data("../documents/input.txt", "\t", np.float64)  # 读取数据
	dataY = loadtxtAndcsv_data("../documents/output.txt", ",", np.float64)

	X = np.array(dataX[:, :], dtype=np.float64)
	y = np.array(dataY[:, :], dtype=np.float64)

	# 归一化操作
	scaler = StandardScaler()
	scaler.fit(X)
	#print("mean:{0}, std:{1}".format(scaler.mean_, scaler.scale_))

	x_train = scaler.transform(X)  # 将训练集归一化
	x_test = scaler.transform(np.array(dataX[:, :]))

	# 线性模型拟合
	model = linear_model.LinearRegression()
	model.fit(x_train, y)

	# 预测结果
	result = model.predict(x_test)
	#print("coef_: {}".format(model.coef_))  # Coefficient of the features 决策函数中的特征系数
	#print("intercept_:{}".format(model.intercept_)) # 又名bias偏置,若设置为False，则为0
	print("result: {}" .format(result))  # 预测结果
	plt.scatter(X[0], y[0], color='green',s = 100,label='real points')
	plt.scatter(X[0], result[0], color='red', marker='v' , label='predicted points')

	# 设置图例
	# plt.legend(loc=(1, 0))
	plt.show()

# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)


# 加载npy文件
def loadnpy_data(fileName):
	return np.load(fileName)

if __name__ == "__main__":
	linearRegression()