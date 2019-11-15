# Linear Regression
import pandas as pd
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)  # set( )设置主题，调色板更常用
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
	1. 加载数据
	2. 标准化
	3. 梯度下降：返回需要的均值，方差，和 theta （目标函数y = theta0 + theta1 * X1 + theta2 * X2）
	4. 预测: 同样先标准化，然后将标准化的X1， X2 与theta相乘求y值
'''
def linearRegression(alpha=0.01, num_iters=5):
	print("加载数据...\n")

	data = loadtxtAndcsv_data(".././documents/data2.txt", ",", np.float64)  # 读取数据
	X = data[:, 0:-1]  # X对应0到倒数第2列
	y = data[:, -1]  # y对应最后一列

	#print("X.shape: {}".format(X.shape)) # (47, 2)
	#print("y.shape: {}".format(y.shape))  # (47,)
	#print("\nX.shap[0]: {}".format(X.shape[0])) # 0表示行，第一维，47
	#print("\nX.shap[1]: {}".format(X.shape[1])) # 1表示列，第二维， 2

	m = len(y)  # 总的数据条数 47
	col = data.shape[1]  # data的列数 3

	X, mu, sigma = featureNormaliza(X)  # 标准化
	plot_X1_X2(X)  # 画图看一下归一化效果

	X = np.hstack((np.ones((m, 1)), X))  # X是真实数据前加了一列1，因为有theta(0)

	print("\n执行梯度下降算法....")

	theta = np.zeros((col, 1))
	y = y.reshape(-1, 1)  # 将行向量转化为列, 二维数组了

	#print("theta.shape: {}".format(theta.shape))  # (3, 1)
	#print("\ny.shape: {}".format(y.shape))  # (47, 1)

	theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

	plotJ(J_history, num_iters)

	return mu, sigma, theta  # 返回均值mu,标准差sigma,和学习的结果theta


# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
	#return pd.read_csv(fileName)
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)


# 加载npy文件
def loadnpy_data(fileName):
	return np.load(fileName)


# 归一化feature
def featureNormaliza(X):
	X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算
	# 定义所需变量
	mu = np.zeros((1, X.shape[1]))
	sigma = np.zeros((1, X.shape[1]))
	# print("mu.shape: {}".format(mu.shape)) # (1, 2)
	# print("\nmu: {}".format(mu))  # [[0 0]]

	mu = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
	sigma = np.std(X_norm, 0)  # 求每一列的标准差
	# print("***********************")
	# print("\nmu: {}".format(mu))  # [2000.68085106    3.17021277]

	for i in range(X.shape[1]):  # 遍历列
		X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # 归一化

	return X_norm, mu, sigma


# 画二维图
def plot_X1_X2(X):
	plt.scatter(X[:, 0], X[:, 1])
	plt.show()


# 梯度下降算法
def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y) # 47
	n = len(theta) # 3

	#print("\nm: {0}, n: {1}: ".format(m,n))
	temp = np.matrix(np.zeros((n, num_iters)))  # 暂存每次迭代计算的theta，转化为矩阵形式

	J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值 400行
	#print("\ntemp: {0}".format(temp.shape))
	#print("\nJ_history: {0}".format(J_history.shape))

	for i in range(num_iters):  # 遍历迭代次数
		h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
		#print("\nh: {0}, x: {1}, theta{2}".format(h.shape, X.shape, theta.shape))
		temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))  # 梯度的计算

		# print("theta: {}".format(theta))
		# print("-:{}".format(theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))))
		# print("temp:{}".format(temp))

		theta = temp[:, i]
		# print("theta: {}".format(theta))
		J_history[i] = computerCost(X, y, theta)  # 调用计算均值方差
		# print('************')

	return theta, J_history


# 计算代价函数（计算出预测-误差的值）的平方
def computerCost(X, y, theta):
	m = len(y)
	J = 0

	J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)  # 计算代价J
	return J


# 画每次迭代代价的变化图
def plotJ(J_history, num_iters):
	x = np.arange(1, num_iters + 1)
	plt.plot(x, J_history)
	plt.xlabel("iterations")
	plt.ylabel("cost")
	plt.title("The cost varies with the number of iterations")
	plt.show()


# 测试linearRegression函数
def testLinearRegression():
	mu, sigma, theta = linearRegression(0.01, 400)
	print("\n计算的theta值为：\n",theta)
	print("\n预测结果为：%f"% predict(mu, sigma, theta))


# 测试学习效果（预测）
def predict(mu, sigma, theta):
	result = 0
	# 注意归一化
	predict = np.array([1650, 3])
	norm_predict = (predict - mu) / sigma
	final_predict = np.hstack((np.ones((1)), norm_predict))

	result = np.dot(final_predict, theta)  # 预测结果
	return result


if __name__ == "__main__":
	#linearRegression(0.01, )
	testLinearRegression()