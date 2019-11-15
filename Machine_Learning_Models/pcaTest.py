# -*- coding:utf-8 -*-
# https://www.jianshu.com/p/4528aaa6dc48
# https://toutiao.io/posts/e38lf1/preview
'''
PCA分析，只根据输入数据的特征进行主成分分析，与输出有多少类型，每个数据对应哪个类型无关。考虑与聚类的结合，PCA
与聚类结果没有直接的关系。原因：PCA只是降低维度，簇并不一定与维度绑定，PCA 的作用顶多就是去掉噪音，减少计算量，并不会剔除簇信息。
'''
import numpy as np
np.set_printoptions(suppress=True)
import os
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import datetime
import time

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


# y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ") # (6400, 64)
# y_train = np.reshape(y_train, (-1, 4096))
#
# x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
# x_train = np.reshape(x_train, (-1, 4096))
# x_train = x_train[:100,:]
# y_train = y_train[:100,:]
#
# print(y_train.shape, x_train.shape) #558 * 4096
# x_train,x_test, y_train, y_test	 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0) # 446, 112
# print("x_train.shape: {}, x_test.shape: {}".format(x_train.shape, x_test.shape))

##Python实现PCA
import numpy as np
import matplotlib.pyplot as plt

def pcaOriginal(x_train,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = x_train.shape
  mean=np.array([np.mean(x_train[:,i]) for i in range(n_features)])

  #normalization
  norm_X=x_train-mean

  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)

  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]

  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(key=lambda x: x[0])
  eig_pairs.reverse()

  # print('Eigenvalues in descending order:')
  # cnt = 0
  # for i in eig_pairs:
  #     if cnt<= 10:
  #       print(i[0])
  #       cnt += 1
  #     else:
  #       break

  tot = sum(eig_val)  # 所有特征值的和
  var_exp = [(i / tot) * 100 for i in sorted(eig_val, reverse=True)]  # 每个特征值的百分比
  # cum_var_exp = np.cumsum(var_exp)

  # print('var_exp in descending order:')
  # print(var_exp[0:10])

  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  print("feature.shape: {}".format(feature.shape))
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  # print(data.shape)
  # print('\nOriginal Method:\n')
  # print(data[1:5, :2].real)
  return data


def pcaSklearn(data, k):
    pca = PCA(n_components=512)
    data = pca.fit_transform(data)  # fit the model

    print('\npcaSklearn Method:\n')
    print(data[1:5, :2])
    return data
    # print('\nMethod Sklearn: PCA by Scikit-learn:')
    # print(pca.transform(y_train[1:10, :]))  # transformed data
    # pca = PCA(n_components=0.95)
    # pca.fit(y_train)
    # print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)
    # print('pca.explained_variance_: ', pca.explained_variance_)
    # print('pca.n_components_: ', pca.n_components_)

def savetxt():
    np.savetxt("x_train.txt", x_train[0:100,:5], fmt="%.6f", delimiter=",")
    np.savetxt("x_train2.txt", x_train[100:200, :5], fmt="%.6f", delimiter=",")
    np.savetxt("x_train3.txt", x_train[200:300,:5], fmt="%.6f", delimiter=",")
    np.savetxt("x_train4.txt", x_train[300:400, :5], fmt="%.6f", delimiter=",")
    np.savetxt("x_train5.txt", x_train[400:500,:5], fmt="%.6f", delimiter=",")

def pcaOriginal(x_train,k):#k is the components you want
  n_samples, n_features = x_train.shape
  mean=np.array([np.mean(x_train[:,i]) for i in range(n_features)])

  #normalization
  norm_X=x_train-mean

  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)

  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  print("len(eig_val): {}".format(len(eig_val)))

  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(key=lambda x: x[0])
  eig_pairs.reverse()

  tot = sum(eig_val)  # 所有特征值的和
  var_exp = [(i / tot) * 100 for i in sorted(eig_val, reverse=True)]  # 每个特征值的百分比

  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  print("feature.shape: {}".format(feature.shape))

  data=np.dot(norm_X,np.transpose(feature))
  return data

if __name__ == '__main__':
    # savetxt()
    starttime = datetime.datetime.now()
    y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ")  # (6400, 64)
    y_train = np.reshape(y_train, (-1, 4096))

    x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
    x_train = np.reshape(x_train, (-1, 4096))
    x_train = x_train[:100, :1024]
    y_train = y_train[:100, :1024]

    print(y_train.shape, x_train.shape)
    print("*************readTime: {}s *************".format(str((datetime.datetime.now() - starttime).seconds)))

    x = pcaOriginal(x_train, 256)
    endtime = datetime.datetime.now()
    print("************* x_train runTime: {}s *************".format(str((endtime - starttime).seconds)))

    y = pcaOriginal(y_train, 256)
    endtime2 = datetime.datetime.now()
    print("************* total runTime: {}s *************".format(str((endtime2 - starttime).seconds)))
    print(x.shape, y.shape)
    # pcaSklearn(x_train, 512)
