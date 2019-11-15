# -*- coding:utf-8 -*-
import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import datetime
import glob
from contextlib import contextmanager

@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

loss_vec = []


# load txt or csv
def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def generate_batches(file_path, allFiles, split):
    allFiles = glob.glob(file_path + allFiles)
    list = []

    for f in allFiles:
        f = loadtxtAndcsv_data(f, split, np.float64)
        list.append(f)

    # concatenate the arrays
    X_Train = np.concatenate(list)
    return X_Train


# y_mean: 3.96254398044, max: 119989.68308, min: 50010.873225    (25 ~ 40)
# y_mean: 80462.7156926, max: 97499.491667, min: 64013.917728    (30 ~ 35)
# y_train.mean: 49879.05646921431, max: 74989.514925, min: 30042.140535 (15 ~ 25)
# y_mean: 74848.01047705958, max: 104761.259505, min: 50011.518975 (25 ~ 35)


# print("y_train.shape: {}".format(y_train.shape)) # (20, 4096)
# print("x_train.mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train), np.min(x_train)))
# print("y_train.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))

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

def create_NewSingleArrayY(y_train):
    for i in range(20):
        a = random_list(600, 3000, 4096)
        y_train[i] = y_train[i] + a

    y_train = np.where(y_train > 50011.518975, y_train, 50011.518975) # min
    y_train = np.where(y_train < 104761.259505, y_train, 104761.259505)  # max
    return y_train

def create_NewSingleArrayX(x_train):
    for i in range(20):
        a = random_list(0.01, 0.05, 4096)
        x_train[i] = x_train[i] + a
    
    x_train = np.where(x_train > -3.4970684027844854, x_train, -3.4970684027844854)  # min
    x_train = np.where(x_train < 3.4164963645508712, x_train, 3.4164963645508712)  # max
    return x_train

def create_FalseArrayY(y_train):
    for i in range(100):
        tmp = create_NewSingleArrayY(y_train[:20, :].copy())  # (20, 4096)
        y_train = np.concatenate((y_train, tmp), axis=0)


    return y_train[20:, :]

def create_FalseArrayX(x_train):
    for i in range(100):
        tmp = create_NewSingleArrayX(x_train[:20,:].copy()) # (20, 4096)
        x_train = np.concatenate((x_train, tmp), axis=0)

    return x_train[20:,:]


if __name__ == '__main__':
    y_train = generate_batches("../documents/900_input/", "/newInput_*.txt", " ")  # 677
    y_train = np.reshape(y_train, (-1, 4096))

    x_train = generate_batches("../documents/900_output/", "/newOutput_64_*.txt", ",")
    x_train = np.reshape(x_train, (-1, 4096))
    print("start x_train.shape: {}".format(x_train.shape)) # (20, 4096)

    scalarX = StandardScaler().fit(x_train)
    x_train = scalarX.transform(x_train)
    print("after StandardScaler, x_train.shape: {}".format(x_train.shape))
    print("before x_real.mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train),np.min(x_train)))


    x_train_false = create_FalseArrayX(x_train)
    print("now x_false.shape: {}".format(x_train_false.shape)) # (1800, 4096)
    # print("now x_false.mean: {}, max: {}, min: {}".format(np.mean(x_train_false), np.max(x_train_false), np.min(x_train_false)))

    x_train = np.concatenate((x_train_false[1400:1980,:], x_train), axis=0) # 350
    x_train_false = x_train_false[0:1400,:]
    print("x_train_concentrate.shape: {}, x_train_false.shape: {}".format(x_train.shape, x_train_false.shape))
    # pca = PCA(n_components=0.95)
    # pca.fit(x_train_false)
    # print('x_train.n_components_: ', pca.n_components_)

    '''
    pca = PCA(n_components=512)
    x_train = pca.fit_transform(x_train)
    x_train_false = pca.transform(x_train_false)
    print("Reduced x_train_concentrate.shape: {}, x_train_false.shape: {}".format(x_train.shape, x_train_false.shape))
    print("**********************************************************************")

    print("Reduced x_real.mean: {}, max: {}, min: {}".format(np.mean(x_train),np.max(x_train), np.min(x_train)))
    print("Reduced x_false.mean: {}, max: {}, min: {}".format(np.mean(x_train_false), np.max(x_train_false), np.min(x_train_false)))
    '''

    print("************************************************************")
    print("before y_real.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))

    y_train_false = create_FalseArrayY(y_train)
    print("now y_false.shape: {}".format(y_train_false.shape)) # (21480, 4096)
    print("now y_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false), np.max(y_train_false), np.min(y_train_false)))
    # print("now y_false[0].mean: {}, max: {}, min: {}".format(np.mean(y_train_false[0:1,:]), np.max(y_train_false[0:1,:]),np.min(y_train_false[0:1,:])))
    # print("now y_false[2].mean: {}, max: {}, min: {}".format(np.mean(y_train_false[2:3, :]), np.max(y_train_false[2:3, :]),np.min(y_train_false[2:3, :])))

    # print(y_train_false[0:2,0:5])
    # print("************************************************************")
    # print(y_train_false[2:4,0:5])

    y_train = np.concatenate((y_train_false[1400:1980, :], y_train), axis=0) # uese for y_real when evaluation
    y_train_false = y_train_false[0:1400, :] # # uese for train

    print("After split 20000, y_train.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))
    # y_train.shape: (2000, 4096), y_train_false.shape: (20000, 4096)
    print("*************************** No reduced! No standarlization! Mean should same ***************************")
    print("train_use y_train_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false), np.max(y_train_false), np.min(y_train_false)))
    print("evaluation_use y_train.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    np.savetxt("y4096_real.txt", y_train, fmt="%.30f", delimiter=",")

    '''
    pcaY = PCA(n_components=512)
    y_train = pcaY.fit_transform(y_train)
    y_train_false = pcaY.transform(y_train_false)
    print("Reduced y_train_concentrate.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))

    print("Reduced y_real.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    print("Reduced y_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false),np.max(y_train_false), np.min(y_train_false)))
    '''

    np.savetxt("x_false_4096.txt", x_train_false, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421
    np.savetxt("y_false_4096.txt", y_train_false, fmt="%.30f", delimiter=",")


    np.savetxt("x_real_4096.txt", x_train, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421

    print("saving finished")
    '''
    xx = loadtxtAndcsv_data("x_false_512_MLK.txt", ",", np.float64)
    yy = loadtxtAndcsv_data("y_false_512_MLK.txt", ",", np.float64)

    x_test = loadtxtAndcsv_data("x_real_512_MLK.txt", ",", np.float64)


    print("x_false.mean: {}, max: {}, min: {}".format(np.mean(xx) , np.max(xx),np.min(xx)))
    print("x_real.mean: {}, max: {}, min: {}".format(np.mean(x_test), np.max(x_test), np.min(x_test)))
    print("********************************************************************************************")
    print("y_false.mean: {}, max: {}, min: {}".format(np.mean(yy), np.max(yy), np.min(yy)))
    '''

