# -*- coding:utf-8 -*-
import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import load_model,Model
from sklearn.decomposition import PCA
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
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

#y_train1 = generate_batches("./documents/MLK_Input/", "/newInput_*.txt", " ") # 677
#y_train1 = np.reshape(y_train, (-1, 4096))
# #
#x_train1 = generate_batches("./documents/MLK_Output/", "/newOutput_64_*.txt", ",")
#x_train1 = np.reshape(x_train, (-1, 4096))
#x mean: 2438.203516746279, max: 2638.268392, y_train.min: 2259.801443
#y mean: 81093.96254398044, max: 119989.68308, y_train.min: 50010.873225
y_train = generate_batches("./documents/500_input/", "/newInput_*.txt", " ") # 677
y_train = np.reshape(y_train, (-1, 4096))

x_train = generate_batches("./documents/500_output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))


print("y_train.shape: {}".format(y_train.shape)) # (179, 4096)
# print("x_train1.mean: {}, max: {}, y_train.min: {}".format(np.mean(x_train), np.max(x_train), np.min(x_train)))
print("y_train.mean: {}, max: {}, y_train.min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train))) # y_train.max: 191998.998976, y_train.min: 2000.197103

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

def createArray(y_train_first):
    a = random_list(500, 1000, 4096)
    a2 = random_list(500, 1000, 4096)
    a3 = random_list(500, 1000, 4096)
    a4 = random_list(500, 1000, 4096)
    a5 = random_list(500, 1000, 4096)
    a6 = random_list(500, 1000, 4096)
    a7 = random_list(500, 1000, 4096)
    a8 = random_list(500, 1000, 4096)
    a9 = random_list(500, 1000, 4096)
    a10 = random_list(500, 1000, 4096)
    a11 = random_list(500, 1000, 4096)
    a12 = random_list(500, 1000, 4096)
    a13 = random_list(500, 1000, 4096)
    a14 = random_list(500, 1000, 4096)
    a15 = random_list(500, 1000, 4096)
    a16 = random_list(500, 1000, 4096)
    a17 = random_list(500, 1000, 4096)
    a18 = random_list(500, 1000, 4096)
    a19 = random_list(500, 1000, 4096)
    a20 = random_list(500, 1000, 4096)

    # print("a.shape: {}".format(a.shape))
    falseArray1 = y_train_first + a
    falseArray2 = y_train_first + a2
    falseArray3 = y_train_first + a3
    falseArray4 = y_train_first + a4
    falseArray5 = y_train_first + a5
    falseArray6 = y_train_first + a6
    falseArray7 = y_train_first + a7
    falseArray8 = y_train_first + a8
    falseArray9 = y_train_first + a9
    falseArray10 = y_train_first + a10
    falseArray11 = y_train_first + a11
    falseArray12 = y_train_first + a12
    falseArray13 = y_train_first + a13
    falseArray14 = y_train_first + a14
    falseArray15 = y_train_first + a15
    falseArray16 = y_train_first + a16
    falseArray17 = y_train_first + a17
    falseArray18 = y_train_first + a18
    falseArray19 = y_train_first + a19
    falseArray20 = y_train_first + a20


    y_train_first = np.concatenate((falseArray1, falseArray2, falseArray3, falseArray4, falseArray5, falseArray6, falseArray7,falseArray8, falseArray9, falseArray10,
                                    falseArray11, falseArray12, falseArray13, falseArray14, falseArray15, falseArray16,falseArray17, falseArray18, falseArray19, falseArray20), axis=0)

    y_train_first = np.where(y_train_first > 50010.873225, y_train_first, 50010.873225)
    return y_train_first

def createX(x_train_first):
    a1 = random_list(-0.05,0.05, 4096)
    a2 = random_list(-0.05,0.05, 4096)
    a3 = random_list(-0.05,0.05, 4096)
    a4 = random_list(-0.05,0.05, 4096)
    a5 = random_list(-0.05,0.05, 4096)
    a6 = random_list(-0.05,0.05, 4096)
    a7 = random_list(-0.05,0.05, 4096)
    a8 = random_list(-0.05,0.05, 4096)
    a9 = random_list(-0.05,0.05, 4096)
    a10 = random_list(-0.05,0.05, 4096)
    a11 = random_list(-0.05, 0.05, 4096)
    a12 = random_list(-0.05, 0.05, 4096)
    a13 = random_list(-0.05, 0.05, 4096)
    a14 = random_list(-0.05, 0.05, 4096)
    a15 = random_list(-0.05,0.05, 4096)
    a16 = random_list(-0.05, 0.05, 4096)
    a17 = random_list(-0.05, 0.05, 4096)
    a18 = random_list(-0.05, 0.05, 4096)
    a19 = random_list(-0.05, 0.05, 4096)
    a20 = random_list(-0.05, 0.05, 4096)

    falseArray1 = x_train_first + a1
    falseArray2 = x_train_first + a2
    falseArray3 = x_train_first + a3
    falseArray4 = x_train_first + a4
    falseArray5 = x_train_first + a5
    falseArray6 = x_train_first + a6
    falseArray7 = x_train_first + a7
    falseArray8 = x_train_first + a8
    falseArray9 = x_train_first + a9
    falseArray10 = x_train_first + a10
    falseArray11 = x_train_first + a11
    falseArray12 = x_train_first + a12
    falseArray13 = x_train_first + a13
    falseArray14 = x_train_first + a14
    falseArray15 = x_train_first + a15
    falseArray16 = x_train_first + a16
    falseArray17 = x_train_first + a17
    falseArray18 = x_train_first + a18
    falseArray19 = x_train_first + a19
    falseArray20 = x_train_first + a20


    x_train_first = np.concatenate((falseArray1, falseArray2, falseArray3, falseArray4, falseArray5, falseArray6, falseArray7,falseArray8, falseArray9, falseArray10,
                                    falseArray11, falseArray12, falseArray13, falseArray14, falseArray15, falseArray16,falseArray17, falseArray18, falseArray19, falseArray20), axis=0)
    # x_train_first = np.where(x_train_first > 0, x_train_first, 0)
    return x_train_first



if __name__ == '__main__':
    print("before x_real.shape: {}".format(x_train.shape))
    scalarX = StandardScaler().fit(x_train)
    x_train = scalarX.transform(x_train)
    print("before x_real.mean: {}, max: {}, min: {}".format(np.mean(x_train), np.max(x_train),np.min(x_train)))

    x1 = createX(x_train)
    x2 = createX(x_train)
    x3 = createX(x_train)
    x4 = createX(x_train)
    x5 = createX(x_train)
    x6 = createX(x_train)
    x7 = createX(x_train)
    x8 = createX(x_train)
    x9 = createX(x_train)
    x10 = createX(x_train)


    x_train_false = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8,x9,x10), axis=0)
    print("now x_false.shape: {}".format(x_train_false.shape)) # (25060, 4096)
    print("now x_false.mean: {}, max: {}, min: {}".format(np.mean(x_train_false), np.max(x_train_false), np.min(x_train_false)))
    
    #pca = PCA(n_components=512)
    #x_train = pca.fit_transform(x_train)

    x_train = np.concatenate((x_train_false[30000:32821,:], x_train), axis=0)
    x_train_false = x_train_false[0:30000,:]
    print("x_train_concentrate.shape: {}, x_train_false.shape: {}".format(x_train.shape, x_train_false.shape))
    #pca = PCA(n_components=0.95)
    #pca.fit(x_train_false)
    #print('x_train.n_components_: ', pca.n_components_)

    pca = PCA(n_components=512)
    x_train = pca.fit_transform(x_train)
    x_train_false = pca.transform(x_train_false)
    print("Reduced x_train_concentrate.shape: {}, x_train_false.shape: {}".format(x_train.shape, x_train_false.shape))
    print("x_real_reduced.shape: {}".format(x_train.shape))
    print("**********************************************************************")
    #
    print("Reduced x_real.mean: {}, max: {}, min: {}".format(np.mean(x_train),np.max(x_train), np.min(x_train)))
    print("Reduced x_false.mean: {}, max: {}, min: {}".format(np.mean(x_train_false), np.max(x_train_false), np.min(x_train_false)))

    print("************************************************************")
    
    print("before y_real.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))

    y1 = createArray(y_train)
    y2 = createArray(y_train)
    y3 = createArray(y_train)
    y4 = createArray(y_train)
    y5 = createArray(y_train)
    y6 = createArray(y_train)
    y7 = createArray(y_train)
    y8 = createArray(y_train)
    y9 = createArray(y_train)
    y10 = createArray(y_train)

    y_train_false = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8,y9, y10), axis=0)
    print("now y_false.shape: {}".format(y_train_false.shape)) # (21480, 4096)
    print("now y_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false), np.max(y_train_false), np.min(y_train_false)))

    #  split (25060, 4096) to (0:20000) and (20000:21821)
    y_train = np.concatenate((y_train_false[30000:32821, :], y_train), axis=0) # uese for y_real when evaluation
    y_train_false = y_train_false[0:30000, :] # # uese for train

    print("After split 35800, y_train.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))
    # y_train.shape: (2000, 4096), y_train_false.shape: (20000, 4096)
    print("*************************** No reduced! No standarlization! Mean should same ***************************")
    print("train_use y_train_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false), np.max(y_train_false), np.min(y_train_false)))
    print("evaluation_use y_train.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    np.savetxt("y512_3w_4096.txt", y_train, fmt="%.30f", delimiter=",")
#    np.savetxt("y_2w_4096.txt", y_train_false, fmt="%.30f", delimiter=",")
    #scalarY = StandardScaler().fit(y_train_false)
    #y_train = scalarY.transform(y_train) # 2千
    #y_train_false = scalarY.transform(y_train_false) # 2万

    #print("*************************** No reduced! Only standarlization. ***************************")
    #print("train_use y_train_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false), np.max(y_train_false), np.min(y_train_false)))
    #print("evaluation_use y_train.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train),np.min(y_train)))
    pcaY = PCA(n_components=512)
    y_train = pcaY.fit_transform(y_train)
    y_train_false = pcaY.transform(y_train_false)
    print("Reduced y_train_concentrate.shape: {}, y_train_false.shape: {}".format(y_train.shape, y_train_false.shape))
    #
    print("Reduced y_real.mean: {}, max: {}, min: {}".format(np.mean(y_train), np.max(y_train), np.min(y_train)))
    print("Reduced y_false.mean: {}, max: {}, min: {}".format(np.mean(y_train_false),np.max(y_train_false), np.min(y_train_false)))

    np.savetxt("x_false_512_3w.txt", x_train_false, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421
    np.savetxt("y_false_512_3w.txt", y_train_false, fmt="%.30f", delimiter=",")
    #
    #
    np.savetxt("x_real_512_3w.txt", x_train, fmt="%.30f", delimiter=",")  # 0:114 && 421:446 && 644:677 # 446:644 # 115:421
    #np.savetxt("y_real_1024_un.txt", y_train, fmt="%.30f", delimiter=",")

    print("saving finished")

    xx = loadtxtAndcsv_data("x_false_512_3w.txt", ",", np.float64)
    yy = loadtxtAndcsv_data("y_false_512_3w.txt", ",", np.float64)
    
    x_test = loadtxtAndcsv_data("x_real_512_3w.txt", ",", np.float64)
   
    
    print("x_false.mean: {}, max: {}, min: {}".format(np.mean(xx) , np.max(xx),np.min(xx)))
    print("x_real.mean: {}, max: {}, min: {}".format(np.mean(x_test), np.max(x_test), np.min(x_test)))
    print("********************************************************************************************")
    print("y_false.mean: {}, max: {}, min: {}".format(np.mean(yy), np.max(yy), np.min(yy)))
