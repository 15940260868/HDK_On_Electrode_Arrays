# -*- coding:utf-8 -*-
'''
PCA分析，只根据输入数据的特征进行主成分分析，与输出有多少类型，每个数据对应哪个类型无关。考虑与聚类的结合，PCA
与聚类结果没有直接的关系。原因：PCA只是降低维度，簇并不一定与维度绑定，PCA 的作用顶多就是去掉噪音，减少计算量，并不会剔除簇信息。
'''
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

path = '/Users/wangxinying/Desktop/Project/Pycharm/dataProcess/Machine_Learning/variance.txt'
# for file in os.listdir(path):
#     if os.path.isfile(os.path.join(path,file))==True:
#             newname=''+file
#             os.rename(os.path.join(path,file),os.path.join(path,newname))
            #print(file,'ok')

def generate_batches(file_path, allFiles, split):
    allFiles = glob.glob(file_path + allFiles)
    list = []

    for f in allFiles:
        f = loadtxtAndcsv_data(f, split, np.float64)
        list.append(f)

    # 把数组联合在一起
    X_Train = np.concatenate(list)
    return X_Train

def loadtxtAndcsv_data(fileName: object, split: object, dataType: object) -> object:
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

var_list = []
std_list = []

y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ") # (6400, 64)
y_train = np.reshape(y_train, (-1, 4096))

x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))
scalarX = StandardScaler().fit(x_train)
x_train = scalarX.transform(x_train)
# print(y_train.shape, x_train.shape)
#x_train,x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0) # 446, 112
#print("x_train.shape: {}, x_test.shape: {}".format(x_train.shape, x_test.shape)) # 677 = 541 + 136

def mean(x_train):
    arr_mean = np.mean(x_train, axis=0)
    arr_var = np.var(x_train, axis=0)
    arr_std = np.std(x_train, axis=0)
    #arr_std = np.std(arr, ddof=1)

    for i in range(len(arr_var)):
        var_list.append(arr_var[i])
        std_list.append(arr_std[i])


    plt.plot(var_list)
    plt.title("var", fontsize=24)
    plt.show()

    plt.plot(std_list)
    plt.title("std", fontsize=24)
    plt.show()

    '''
    fileObject = open(path, 'w')
    for ip in range(len(arr_var)):
        var_list.append(arr_var[ip])
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    '''

def pcaTest(y_train,y_test):
    pca = PCA(n_components=0.95)
    pca.fit(y_train)
    print('pca.n_components_: ', pca.n_components_)
    # print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)
    # print('pca.explained_variance_: ', pca.explained_variance_)


    # print("****** 3D ********")
    # pca = PCA(n_components=3)
    # x_train = pca.fit_transform(x_train)
    # print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)
    # print('pca.explained_variance_: ', pca.explained_variance_)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], marker='o')
    # plt.show()

    #print("**************")
    pca = PCA(n_components=5)
    y_train = pca.fit_transform(y_train) # 541 * 5
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)

    ax1.plot(y_train[:,0:1])
    ax2.plot(y_train[:, 1:2])
    ax3.plot(y_train[:, 2:3])
    ax4.plot(y_train[:, 3:4])
    ax5.plot(y_train[:, 4:5])
    #plt.show()
    [row, cols] = y_train.shape
    [rowT, colsT] = y_test.shape
    np.savetxt("x_train.txt", y_train, fmt="%.6f", delimiter=",")
    np.savetxt("x_test.txt", y_train, fmt="%.6f", delimiter=",")
    np.savetxt("y_train.txt", y_train, fmt="%.6f", delimiter=",")
    np.savetxt("y_test.txt", y_train, fmt="%.6f", delimiter=",")
    print(np.mean(y_train),np.max(y_train), np.min(y_train))  # -4.579146115484608e-11 2155438.596736644 -1755596.0732554172
    print("***************************")
    print(np.mean(y_train[:,0:1]),np.max(y_train[:,0:1]),np.min(y_train[:,0:1]))
    print(np.mean(y_train[:, 1:2]), np.max(y_train[:, 1:2]), np.min(y_train[:, 1:2]))
    print(np.mean(y_train[:,2:3]),np.max(y_train[:,2:3]),np.min(y_train[:,2:3]))
    print(np.mean(y_train[:, 3:4]), np.max(y_train[:, 3:4]), np.min(y_train[:, 3:4]))
    print(np.mean(y_train[:, 4:5]), np.max(y_train[:, 4:5]), np.min(y_train[:, 4:5]))
    print("***************************")

    ratio = pca.explained_variance_ratio_
    weights = []
    for i in range(len(ratio)):
        weights.append(ratio[i]/np.sum(ratio))
    print(weights)

def processX(x_train):
    # pca = PCA(n_components=0.95)
    # pca.fit(x_train)
    #print('pca.n_components_: ', pca.n_components_)
    print(x_train.shape)
    print("before: {}".format(x_train[1,:5]))
    model = PCA(n_components=336).fit(x_train)  # (677, 4096)
    x_reduced = model.transform(x_train)
    print("after dimensionality reduction: {}".format(x_reduced.shape))

    # Ureduce = model.components_  # 得到降维用的Ureduce
    # x_pre = np.dot(Z, Ureduce)  # 数据恢复
    # print("recover x_pre.shape: {}".format(x_pre.shape))
    # print("recover: {}".format(x_pre[1, :5]))
    X_recovered = model.inverse_transform(x_reduced)
    print("recover X_recovered.shape: {}".format(X_recovered.shape))
    print("recover: {}".format(X_recovered[1, :5]))
    [rows, cols] = x_train.shape
    loss = []
    print("x_train.shape: {}".format(x_train.shape))
    for i in range(rows):
        for j in range(cols):
            loss.append(abs(x_train[i][j] - X_recovered[i][j]))
            # if abs(x_train[i][j] - X_recovered[i][j]) > 500:
            #     print(x_train[i][j] , X_recovered[i][j])
    print("mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train), np.min(x_train)))
    print("mean: {}, max: {}, min: {}".format(np.mean(loss), np.max(loss), np.min(loss)))

def processY():
    print("before: {}".format(y_train[1,:5]))
    model = PCA(n_components=247).fit(y_train)  # (541, 4096)
    y_reduced = model.transform(y_train)
    print("after dimensionality reduction: {}".format(y_reduced.shape))

    y_recovered = model.inverse_transform(y_reduced)
    print("recover y_recovered.shape: {}".format(y_recovered.shape))
    print("recover: {}".format(y_recovered[1, :5]))
    [rows, cols] = y_train.shape
    loss = []
    print("x_train.shape: {}".format(y_train.shape))
    for i in range(rows):
        for j in range(cols):
            loss.append(abs(y_train[i][j] - y_recovered[i][j]))

    print("mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    print("mean: {}, max: {}, min: {}".format(np.mean(loss), np.max(loss), np.min(loss)))

if __name__ == '__main__':
    x_test = generate_batches("../documents/test_output/", "/newOutput_64_*.txt", ",")
    x_test = np.reshape(x_test, (-1, 4096))
    processX(x_test)

    # y_test -= np.mean(y_test, axis=0)  # 减去均值，使得以0为中心
    # y_test /= np.std(y_test, axis=0)  # 标准化
    # pcaTest(y_train,y_test)


# if __name__ == '__main__':
#     for i in range(89):
#         data = loadtxtAndcsv_data("../documents/test_output/Output_64_%d.txt" % (i+700), ",", np.float64)
#         processed = np.reshape(np.array(data), (64, 64))
#         np.savetxt("../documents/test_output/newOutput_64_%d.txt" % (i+700), processed, fmt="%.6f", delimiter=",")
