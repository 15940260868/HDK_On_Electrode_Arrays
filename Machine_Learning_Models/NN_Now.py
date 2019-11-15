# -*- coding:utf-8 -*-
import numpy as np

import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from keras import optimizers

from sklearn.model_selection import StratifiedKFold
import datetime
import glob

loss_vec = []

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def generate_batches(file_path, allFiles, split):
    allFiles = glob.glob(file_path + allFiles)
    list = []

    for f in allFiles:
        f = loadtxtAndcsv_data(f, split, np.float64)
        list.append(f)

    X_Train = np.concatenate(list)
    return X_Train

# (137, 128) # #(308,256) # (197,164) # (100, 84)

#(308,256)
def baseline_model2():
    model = Sequential()
    model.add(Dense(512*4, input_dim=512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512*4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal'))
    # loss = tf.losses.huber_loss(delta=1.0)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model

def train(x_train, y_train):
    model = baseline_model2()
    hist = model.fit(x_train, y_train, epochs=20, batch_size=100)
    model.save('ec2_512_un.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
    del model  # deletes the existing model


loss_vec = []

def evaluteModel(x_test, loss_vec):
    y_real = loadtxtAndcsv_data("y512_real_MLK.txt", ",", np.float64)
    y_real = np.reshape(y_real, (-1, 4096))
    y_test = y_real

    pca = PCA(n_components=512)
    y_real = pca.fit_transform(y_real)

    model = load_model('ec2_512_un.h5')
    y_predict =  model.predict(x_test) # # 预测出来的被标准化的512
    y_predict = pca.inverse_transform(y_predict) # 预测出来的512还原到4096

    [rows, cols] = y_test.shape

    for i in range(rows):
        cnt = 0
        for j in range(cols):
            loss = abs(abs(y_predict[i][j]) - abs(y_test[i][j])) / abs(y_test[i][j])
            loss_vec.append(loss)


    print("loss_vec.mean: {}, max: {}, min: {}".format(np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))
    return loss_vec


seed = 7
from sklearn.model_selection import KFold
if __name__ == '__main__':
    starttime = datetime. datetime.now()
    X = loadtxtAndcsv_data("x_false_512_MLK.txt", ",", np.float64)
    Y = loadtxtAndcsv_data("y_false_512_MLK.txt", ",", np.float64)

    x_test = loadtxtAndcsv_data("x_real_512_MLK.txt", ",", np.float64)

    X = X[0:1400, :]
    Y = Y[0:1400, :]

    # print("x_train: {}. y_train: {}".format(X.shape, Y.shape)) # (1400, 512)

    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    cvscores = []
    for train, test in kfold.split(X, Y):
        print("x_train: {}. y_train: {}".format(X[train].shape, Y[train].shape))
        model = baseline_model2()
        hist = model.fit(X[train], Y[train], epochs=20, batch_size=100)  # Fit the model
        model.save('ec2_512_un.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
        del model  # deletes the existing model

        # evaluate the model
        # scores = model.evaluate(X[test], Y[test], verbose=0)
        loss_vec.clear()
        scores = evaluteModel(x_test, loss_vec)
        loss_arr = np.array(scores)
        loss_arr = np.reshape(loss_arr, (-1, 4096))
        print("loss_arr_512.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr.shape, np.mean(loss_arr),
                                                                          np.max(loss_arr), np.min(loss_arr)))

        cvscores.append(np.mean(loss_arr))
    print(cvscores)
    print("cvscores mean: {}".format(np.mean(cvscores)))


    ##################################################################################
    #                                  Evaluation
    ##################################################################################
    # loss = evaluteModel(x_real_reduced, loss_vec)
    # loss_arr = np.array(loss)
    # loss_arr = np.reshape(loss_arr, (-1, 4096))
    # np.savetxt("ec2_loss_arr_512.txt", loss_arr, fmt="%.6f", delimiter=",")
    #
    # loss_arr2 = loadtxtAndcsv_data("ec2_loss_arr_512.txt", ",", np.float64)
    # print("loss_arr_512.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr2.shape, np.mean(loss_arr2),np.max(loss_arr2), np.min(loss_arr2)))

