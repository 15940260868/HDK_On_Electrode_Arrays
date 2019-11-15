# https://blog.csdn.net/SA14023053/article/details/51703204
# https://vinking934296.iteye.com/blog/2357242
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 导入多项式回归模型
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image
import os

np.set_printoptions(suppress=True)
# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def generate_batches(file_path, allFiles, split):
	allFiles = glob.glob(file_path + allFiles)
	list = []

	for f in allFiles:
		f = loadtxtAndcsv_data(f, split, np.float64)
		list.append(f)

	# 把数组联合在一起
	X_Train = np.concatenate(list)
	return X_Train

linear_loss = []
polynomial_loss = []
KNN_loss = []
tree_loss = []

x_train = loadtxtAndcsv_data("x_false_512_900X.txt", ",", np.float64)
y_train = loadtxtAndcsv_data("y_false_512_900X.txt", ",", np.float64)
x_test = loadtxtAndcsv_data("x_real_512_900X.txt", ",", np.float64)
# y_test = loadtxtAndcsv_data("y512_real_900X.txt", ",", np.float64)

# x_train = x_train[:1000,:]
# y_train = y_train[:1000,:]
print(x_train.shape, x_test.shape)


def performance_metric(y_true, y_predict):
	score = r2_score(y_true, y_predict)
	return score

def virsualTree(model_tree):
	# data_target_name = np.unique(data_["class"])
	# dot_tree = tree.export_graphviz(model_tree, out_file=None, feature_names=data_feature_name,
	#                                 class_names=data_target_name, filled=True, rounded=True, special_characters=True)
	dot_tree = tree.export_graphviz(model_tree, out_file=None, filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_tree)
	img = Image(graph.create_png())
	graph.write_png("out.png")


# 多项式回归
def polynomial():
	print("polynomial trainning ...")

	n_features = 10
	train_X = x_train[:, 0:n_features]
	train_Y = y_train[:, 0:n_features]
	test_X = x_test[:, 0:n_features]
	test_Y = y_test[:, 0:n_features]

	polynomial = PolynomialFeatures(degree=2)  # 二次多项式
	x_transformed = polynomial.fit_transform(train_X)  # x每个数据对应的多项式系数
	# print(x_transformed.shape) (100, 8394753)

	model = LinearRegression()  # 创建回归器
	model.fit(x_transformed, train_Y)  # 训练数据

	xx_transformed = polynomial.transform(test_X)
	predict_Y = model.predict(xx_transformed)

	test_Y.reshape(-1)
	predict_Y.reshape(-1)

	for i in range(len(predict_Y)):
		loss = abs(predict_Y[i] - test_Y[i])
		polynomial_loss.append(loss)

	print("polynomial_y_predict: ", predict_Y)
	print("polynomial:\nmean: {}, max: {}, min: {}".format(np.mean(polynomial_loss), np.max(polynomial_loss),
														   np.min(polynomial_loss)))
	pass

# R-squared value, The mean squared error, The mean absoluate error
def try_different_method(model, lossList):
	X2 = np.hstack([x_train, x_train ** 2])
	X3 = np.hstack([x_test, x_test ** 2])

	model.fit(X2,y_train)

	# virsualTree(model)

	y_predict = model.predict(X3)
	print("y_predict.shape: {}".format(y_predict.shape))

	y_real = loadtxtAndcsv_data("y512_real_900X.txt", ",", np.float64)
	y_real = np.reshape(y_real, (-1, 4096))
	y_test = y_real

	pca = PCA(n_components=512)
	y_real = pca.fit_transform(y_real)
	print("y_real_scala_reduced: {}".format(y_real.shape))
	print(y_real[0:1, 0:5])

	print("y_predict: {}".format(y_predict.shape))
	print(y_predict[0:1, 0:5])
	y_predict = pca.inverse_transform(y_predict)  # 预测出来的512还原到4096
	print("y_predict_recover: {}".format(y_predict.shape))

	[rows, cols] = y_predict.shape
	for i in range(rows):
		for j in range(cols):
			if y_test[i][j] != 0:
				loss = abs(y_predict[i][j] - y_test[i][j]) / y_test[i][j]
				lossList.append(loss)
			else:
				print(y_predict[i][j], y_test[i][j])
				lossList.append(0)

	print("loss_vec.mean: {}, max: {}, min: {}".format(np.mean(lossList), np.max(lossList), np.min(lossList)))
	return lossList


def linear():
	linear_Model = LinearRegression()
	try_different_method(linear_Model, linear_loss)

	print("Linear mean: {}, max: {}, min: {}".format(np.mean(linear_loss), np.max(linear_loss), np.min(linear_loss)))
	print("coef_: {}".format(linear_Model.coef_.shape))
	print(linear_Model.coef_[:1,:])
	print("intercept_:{}".format(len(linear_Model.intercept_)))
	print("intercept_:{}".format(linear_Model.intercept_[0]))


# https://blog.csdn.net/u014727529/article/details/78422538
def decisionTree():

	tree_Model = tree.DecisionTreeRegressor(max_depth=1)
	try_different_method(tree_Model, tree_loss)

	print("DecisionTree mean: {}，max: {}, min: {}".format(np.mean(tree_loss), np.max(tree_loss), np.min(tree_loss)))
	"""
	for i in range(1, 5):
		tree_Model = tree.DecisionTreeRegressor(max_depth=i) # max_depth=(i+1)
		#scores = cross_val_score(tree_Model, x_train, y_train)
		#print("score: {}".format(scores))
		try_different_method(tree_Model, tree_loss)

		print("tree {}: mean: {}, max: {}, min: {}".format(i, np.mean(tree_loss), np.max(tree_loss), np.min(tree_loss)))
		print("***********************************************")
    """
def KNN():
	# SVM_Model = svm.SVR(kernel='linear') # kernel='linear' # kernel='poly' # kernel='rbf'
	KNN_Model = neighbors.KNeighborsRegressor()
	try_different_method(KNN_Model, KNN_loss)

	# multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear')).fit(X, y)
	# multioutputregressor.predict(X)

	# knn = neighbors.KNeighborsRegressor()
	# regr = MultiOutputRegressor(knn)
	# regr.fit(x_train, y_train)
	# y_predict = regr.predict(x_test)
	# print(y_predict)
	#
	# for i in range(len(y_predict)):
	#     loss = abs(y_predict[i] - y_test[i])
	#     KNN_loss.append(loss)

	# mean: 272.4132804820311, max: 1036.5285434000107, min: 1.4551915228366852e-11
	print("SVM:\nmean: {}, max: {}, min: {}".format(np.mean(KNN_loss), np.max(KNN_loss), np.min(KNN_loss)))

# https://stats.stackexchange.com/questions/153853/regression-with-scikit-learn-with-multiple-outputs-svr-or-gbm-possible
def randomForest():
	rf = ensemble.RandomForestRegressor(n_estimators=20)  # 20 decisionTree
	try_different_method(rf,forest_loss)

	# mean: 85.39847382499866, max: 221.62649005001003, min: 9.098004750005202
	print("forest:\nmean: {}, max: {}, min: {}".format(np.mean(forest_loss), np.max(forest_loss), np.min(forest_loss)))



if __name__ == '__main__':
	decisionTree() # 0.07885485480504638，max: 0.3085096772701622, min: 0
	# polynomial()
	# linear() #  0.008384589639006672, max: 0.10716356559154637, min: 0
	# randomForest()


