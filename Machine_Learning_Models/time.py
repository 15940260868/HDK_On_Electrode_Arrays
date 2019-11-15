# -*- coding:utf-8 -*-
import numpy as np
import time
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime
import time
import glob
from contextlib import contextmanager

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

y_train = generate_batches("./documents/900_input/", "/newInput_*.txt", " ") # 677
y_train = np.reshape(y_train, (-1, 4096))

x_train = generate_batches("./documents/900_output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))

# print("y_train.shape: {}".format(y_train.shape)) # (20, 4096)
print("x_train.mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train), np.min(x_train)))
print("y_train.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))

def random_list(min, max, length):
    min, max = (float(min), float(max)) if min <= max else (float(max), float(min))

    random_list = []
    for i in range(int(length / 2)):
        tmp = min+np.round(np.random.rand(),2) * (max-min)
        random_list.append(tmp)
        random_list.append(-tmp)

    random_array = np.array(random_list)
    random_array = np.reshape(random_array, (-1, 4096))
    return random_array

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

  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(key=lambda x: x[0])
  eig_pairs.reverse()

  tot = sum(eig_val)  # 所有特征值的和
  var_exp = [(i / tot) * 100 for i in sorted(eig_val, reverse=True)]  # 每个特征值的百分比

  feature = np.array([ele[1] for ele in eig_pairs[:k]])
  print("feature.shape: {}".format(feature.shape))

  data = np.dot(norm_X, np.transpose(feature))
  return data


def create_NewSingleArrayX(x_train):
    for i in range(20):
        a = random_list(0.01, 0.05, 4096)
        x_train[i] = x_train[i] + a

    x_train = np.where(x_train > -3.4970684027844854, x_train,-3.4970684027844854)  # min
    x_train = np.where(x_train < 3.4164963645508712, x_train, 3.4164963645508712)  # max
    return x_train

def create_NewSingleArrayY(y_train):
    for i in range(20):
        #a = random_list(500, 2500, 4096)
        a = random_list(300, 1500, 4096)
        y_train[i] = y_train[i] + a

    y_train = np.where(y_train > 50011.518975, y_train,50011.518975) # min
    y_train = np.where(y_train < 104761.259505, y_train, 104761.259505)  # max
    return y_train

def create_FalseArrayY(y_train):
    for i in range(25):
        tmp = create_NewSingleArrayY(y_train[:20, :].copy())  # (20, 4096)
        y_train = np.concatenate((y_train, tmp), axis=0)


    return y_train[20:, :]

def create_FalseArrayX(x_train):
    for i in range(25):
        tmp = create_NewSingleArrayX(x_train[:20,:].copy()) # (20, 4096)
        x_train = np.concatenate((x_train, tmp), axis=0)

    return x_train[20:,:]

if __name__ == '__main__':
    starttime = time.time()
    print("before x_real.shape: {}".format(x_train.shape))
    scalarX = StandardScaler().fit(x_train)
    x_train = scalarX.transform(x_train)
    #print("before x_real.mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train), np.min(x_train)))

    x_train_false = create_FalseArrayX(x_train)
    x_train = np.concatenate((x_train_false[400:480,:], x_train), axis=0) # 21821
    x_train_false = x_train_false[0:400,:]
    print("Reduced x_train.shape: {}, x_train_false.shape: {}".format(x_train.shape, x_train_false.shape))

    y_train_false = create_FalseArrayY(y_train)
    y_train = np.concatenate((y_train_false[400:480, :], y_train), axis=0)  # uese for y_real when evaluation
    y_train_false = y_train_false[0:400, :]  # # uese for train

    print("Reduced y_train.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))

    np.savetxt("y512_real_400.txt", y_train, fmt="%.30f", delimiter=",")
    print("Create time: {}".format(time.time() - starttime))

    starttime2 = time.time()
    # pca = PCA(n_components=512)
    # x_train = pca.fit_transform(x_train)
    # x_train_false = pca.transform(x_train_false)

    #pcaY = PCA(n_components=512)
    #y_train = pcaY.fit_transform(y_train)
    #y_train_false = pcaY.transform(y_train_false)

    x_train = pcaOriginal(x_train, 512)
    x_train_false = pcaOriginal(x_train_false, 512)

    y_train = pcaOriginal(y_train,512)
    y_train_false = pcaOriginal(y_train_false,512)
    
    pca5 = PCA(n_components=0.6)
    pca5.fit(y_train_false)
    print('pca_60%.n_components_: ', pca5.n_components_)

    
    pca4 = PCA(n_components=0.8)
    pca4.fit(y_train)
    print('pca_90%.n_components_: ', pca4.n_components_)

    pca3 = PCA(n_components=0.99)
    pca3.fit(y_train)
    print('pca_99%.n_components_: ', pca.n_components_)

    print("PCA time: {}".format(time.time() - starttime2))

    print("Reduced y_train_concentrate.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))

    print("Reduced y_real.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    print("Reduced y_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false),np.max(y_train_false), np.min(y_train_false)))

    np.savetxt("x_false_400.txt", x_train_false, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421
    np.savetxt("y_false_400.txt", y_train_false, fmt="%.30f", delimiter=",")
    np.savetxt("x_real_400.txt", x_train, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421

    print("Savw time: {}".format(time.time() - starttime2))
