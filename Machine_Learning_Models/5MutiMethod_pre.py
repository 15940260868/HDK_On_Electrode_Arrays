# https://blog.csdn.net/SA14023053/article/details/51703204
# https://vinking934296.iteye.com/blog/2357242
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
forest_loss = []

predict_list = []
real_list = []

y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ") # (6400, 64)
y_train = np.reshape(y_train, (-1, 4096)) # 100 * 4096
# y_train = np.argmax(y_train, axis=1)

x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))  # 100 * 4096
# x_train = np.argmax(x_train, axis=1)

x_train,x_test, y_train, y_test	 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0) # 372 93

x_train -= np.mean(x_train, axis=0)  # 减去均值，使得以0为中心
x_train /= np.std(x_train, axis=0)  # 归一化
y_train -= np.mean(y_train, axis=0)  # 减去均值，使得以0为中心
y_train /= np.std(y_train, axis=0)  # 标准化
x_test -= np.mean(x_test, axis=0)  # 减去均值，使得以0为中心
x_test /= np.std(x_test, axis=0)  # 归一化

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def plot(y_predict):
    # plt.scatter(x_test, y_test, color='green', marker='o', label='real points')
    # plt.scatter(x_test, y_test, color='green', marker='o', label='real points')
    plt.scatter(y_test, y_predict, color='red', marker='v', label='x:y_test, y:predicted')
    # plt.scatter(x_train[:, 1], y_train[:, 1], c='r', marker='o')
    plt.legend()
    plt.show()

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
def try_different_method(model, lossList, pca, x_train, x_test):
#def try_different_method(model, lossList):
    pca = PCA(n_components=pca)
    #print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)
    #print('pca.explained_variance_: ', pca.explained_variance_)
    x_train = pca.fit_transform(x_train)
    model.fit(x_train,y_train)

    x_test = pca.transform(x_test)
    y_predict = model.predict(x_test)
    y_predict *= np.std(y_train, axis=0)
    y_predict += np.mean(y_train, axis=0)

    # virsualTree(model)
    #
    # score = model.score(x_test, y_test)
    # print("{0}.score: {1}".format(model, score))
    # print(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(y_predict)))

    [rows, cols] = y_predict.shape

    for i in range(len(rows)):
        for j in range(len(cols)):
            loss = abs(y_predict[i][j] - y_test[i][j])
            lossList.append(loss)

    return lossList

def linear():
    linear_Model = LinearRegression()

    try_different_method(linear_Model, linear_loss, 2, x_train, x_test)
    print("linear:\nmean: {}, max: {}, min: {}".format(np.mean(linear_loss), np.max(linear_loss), np.min(linear_loss)))

    # print("coef_: {}".format(linear_Model.coef_))
    # print("intercept_:{}".format(linear_Model.intercept_))
    # mean: 1.0280860607281284e-09, max: 6.461050361394882e-09, min: 0.0


# https://blog.csdn.net/u014727529/article/details/78422538
def decisionTree():
    tree_Model = tree.DecisionTreeRegressor(max_depth=3)
    try_different_method(tree_Model, tree_loss, 2, x_train, x_test)
    print("decisionTree mean error: {}".format(np.mean(tree_loss)))
    # for i in range(10):
    #     tree_Model = tree.DecisionTreeRegressor(max_depth=3) # max_depth=(i+1)
    #
    #     #scores = cross_val_score(tree_Model, x_train, y_train)
    #     #print("score: {}".format(scores))
    #
    #     #try_different_method(tree_Model, tree_loss)
    #     try_different_method(tree_Model, tree_loss, 100 * (i+1) , x_train, x_test)
    #
    #     print("tree {}: mean: {}, max: {}, min: {}".format(i, np.mean(tree_loss), np.max(tree_loss), np.min(tree_loss)))
    #     tree_mean.append(np.mean(tree_loss))
    #     tree_min.append(np.min(tree_loss))
    #     print("***********************************************")

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
    # test()
    # plot()
    # KNN()
    # SVM()
    decisionTree()
    # polynomial()
    # linear()
    # randomForest()


