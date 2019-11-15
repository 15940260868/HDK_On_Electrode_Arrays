# -*- coding:utf-8 -*-
import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import datetime
import glob

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

    # 把数组联合在一起
    X_Train = np.concatenate(list)
    return X_Train

# (137, 128) # #(308,256) # (197,164) # (100, 84)

#(308,256)
def baseline_model2():
    model = Sequential()
    model.add(Dense(512*8, input_dim=512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512*8, kernel_initializer='normal', activation='relu'))
  
    model.add(Dense(512, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model


def kerasPlot(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# https://blog.csdn.net/u012193416/article/details/79376345
def train(x_train, y_train):
    print('Training -----------')
    model = baseline_model2()

    hist = model.fit(x_train, y_train, epochs=60, batch_size=100)
    # print(hist.history.keys())
    #kerasPlot(hist)
    model.save('ec2_512_un.h5' )  # HDF5 file, you have to pip3 install h5py if don't have it
    del model  # deletes the existing model



def plot(loss_vec):
    plt.plot(loss_vec, 'r--', label='Test Loss')
    plt.ylabel('Loss')
    plt.show()

loss_vec = []
def evaluteModel(x_test, loss_vec):
    y_real = loadtxtAndcsv_data("y512un_2q_4096.txt", ",", np.float64)
    y_real = np.reshape(y_real, (-1, 4096))
    y_test = y_real

    #scalarY = StandardScaler().fit(y_real)
    #y_real = scalarY.transform(y_real) # # 标准化 4096
    #print("y_real_scala: {}".format(y_real.shape))

    pca = PCA(n_components=512)
    y_real = pca.fit_transform(y_real)
    # y_real = model.transform(y_real) # # standardization 336
    print("y_real_scala_reduced: {}".format(y_real.shape))
    print(y_real[0:1,0:5])

    model = load_model('ec2_512_un.h5')
    y_predict =  model.predict(x_test) # # predicted is standardized 512
    print("y_predict: {}".format(y_predict.shape)) 
    print(y_predict[0:1,0:5])
    y_predict = pca.inverse_transform(y_predict) # predicted 512 was recovered to 4096
    print("y_predict_recover: {}".format(y_predict.shape))

    #y_predict = scalarY.inverse_transform(y_predict) # predicted 512 was recovered to 4096， then recovered to the begining of standardization
    #print("y_predict_recover_scala: {}".format(y_predict.shape)) ## original 4096

    [rows, cols] = y_test.shape

    cnt = 0
    for i in range(rows):
        for j in range(cols):
            loss = abs(y_predict[i][j] - y_test[i][j]) / y_test[i][j]
            loss_vec.append(loss)
            if loss > 0.05:
                cnt += 1
                #print(y_predict[i][j], y_test[i][j])

    print("cnt: {}, loss_vec.mean: {}, max: {}, min: {}".format(cnt/4096, np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))
    return loss_vec

if __name__ == '__main__':
    starttime = datetime. datetime.now()
    # x_train = loadtxtAndcsv_data("x_false_512.txt", ",", np.float64)
    # y_train = loadtxtAndcsv_data("y_false_512.txt", ",", np.float64)
    #
    # x_real_reduced = loadtxtAndcsv_data("x_real_512.txt", ",", np.float64)
    #y_real_reduced = loadtxtAndcsv_data("y_real_512_un.txt", ",", np.float64)
    # print(x_train.shape, y_train.shape,x_real_reduced.shape) # (18900, 378) (18900, 378) (378, 378) (378, 378)

    ##################################################################################
    #                                  Evaluation
    ##################################################################################
    # loss = evaluteModel(x_real_reduced, loss_vec)
    # loss_arr = np.array(loss)
    # loss_arr = np.reshape(loss_arr, (-1, 4096))
    # np.savetxt("y_predict.txt", y_predict, fmt="%.6f", delimiter=",")

    ##################################################################################
    #                                  Evaluation
    ##################################################################################
    loss_arr1 = loadtxtAndcsv_data("ec2_DecisionTree_loss_920X.txt", ",", np.float64)
    loss_arr2 = loadtxtAndcsv_data("ec2_loss_arr_512_900X.txt", ",", np.float64)
    loss_arr3 = loadtxtAndcsv_data("ec2_linear_loss_512_900X.txt", ",", np.float64)
    print("NN.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr2.shape, np.mean(loss_arr2), np.max(loss_arr2), np.min(loss_arr2)))
    print("Tree.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr1.shape, np.mean(loss_arr1), np.max(loss_arr1), np.min(loss_arr1)))
    print("Linear.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr3.shape, np.mean(loss_arr3), np.max(loss_arr3),np.min(loss_arr3)))

    print("************************************************************")
    loss_arr4 = loadtxtAndcsv_data("ec2_DecisionTree_loss_512_920X.txt", ",", np.float64)
    loss_arr5 = loadtxtAndcsv_data("ec2_loss_512_920XX.txt", ",", np.float64)
    loss_arr6 = loadtxtAndcsv_data("ec2_linear_loss_512_920X.txt", ",", np.float64)
    print("NN.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr5.shape, np.mean(loss_arr5), np.max(loss_arr5), np.min(loss_arr5)))
    print("Tree.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr4.shape, np.mean(loss_arr4), np.max(loss_arr4), np.min(loss_arr4)))
    print("Linear.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr6.shape, np.mean(loss_arr6), np.max(loss_arr6),np.min(loss_arr6)))


    np.savetxt("n900.txt", loss_arr2.reshape(-1,1), fmt="%.6f", delimiter=",")
    np.savetxt("n900_1.txt",loss_arr2[:1,:].reshape(-1,1), fmt="%.6f", delimiter=",")

    np.savetxt("t900.txt", loss_arr1.reshape(-1, 1), fmt="%.6f", delimiter=",")
    np.savetxt("t900_1.txt", loss_arr1[:1, :].reshape(-1, 1), fmt="%.6f", delimiter=",")

    np.savetxt("l900.txt", loss_arr3.reshape(-1, 1), fmt="%.6f", delimiter=",")
    np.savetxt("l900_1.txt", loss_arr3[:1, :].reshape(-1, 1), fmt="%.6f", delimiter=",")

    np.savetxt("n920.txt", loss_arr5.reshape(-1, 1), fmt="%.6f", delimiter=",")
    np.savetxt("n920_1.txt", loss_arr5[:1, :].reshape(-1, 1), fmt="%.6f", delimiter=",")

    np.savetxt("t920.txt", loss_arr4.reshape(-1, 1), fmt="%.6f", delimiter=",")
    np.savetxt("t920_1.txt", loss_arr4[:1, :].reshape(-1, 1), fmt="%.6f", delimiter=",")

    np.savetxt("l920.txt", loss_arr6.reshape(-1, 1), fmt="%.6f", delimiter=",")
    np.savetxt("l920_1.txt", loss_arr6[:1, :].reshape(-1, 1), fmt="%.6f", delimiter=",")

    '''
    dict = {'row': []}
    [row, col] = loss_arr2.shape
    for i in range(row):
        cnt = 0
        for j in range(col):
            if(loss_arr2[i][j] > 0.05):
                cnt += 1
        dict["row"].append(cnt)

    value, = dict.values()
    ratioCnt = 0
    for i in range(len(value)):
        if value[i] != 0:
            print(i, value[i])
            ratioCnt += 1
    print("ratioCnt:{} ".format(ratioCnt))

    print("************************************************************")
    y_real = loadtxtAndcsv_data("y512_real_900X.txt", ",", np.float64)
    y_real = np.reshape(y_real, (-1, 4096)) # 2000 * 4096

    y_predict = loadtxtAndcsv_data("y_predict_900.txt", ",", np.float64)
    y_predict = np.reshape(y_predict, (-1, 4096))  # 2000 * 4096

    print("y_real.shape: {}, mean: {}, max: {}, min: {}".format(y_real.shape, np.mean(y_real), np.max(y_real), np.min(y_real)))
    print("y_real[0].mean: {}, max: {}, min: {}".format(np.mean(y_real[0:1,:]), np.max(y_real[0:1,:]), np.min(y_real[0:1,:])))
    print("y_real[2].mean: {}, max: {}, min: {}".format(np.mean(y_real[2:3,:]), np.max(y_real[2:3,:]), np.min(y_real[2:3,:])))

    print("************************************************************")
    print("y_predict.shape: {}, mean: {}, max: {}, min: {}".format(y_predict.shape, np.mean(y_predict), np.max(y_predict),np.min(y_predict)))
    print("y_predict[0].mean: {}, max: {}, min: {}".format(np.mean(y_predict[0:1, :]), np.max(y_predict[0:1, :]),np.min(y_predict[0:1, :])))
    print("y_predict[2].mean: {}, max: {}, min: {}".format(np.mean(y_predict[2:3, :]), np.max(y_predict[2:3, :]),np.min(y_predict[2:3, :])))

    print(y_predict[0:2, 0:5])
    print("************************************************************")
    print(y_real[0:2, 0:5])

    list1 = []
    list2 = []
    predct1 = []
    predct2 = []
    loss1 = []
    loss2 = []

    # 2, 3, 7, 10, 12, 16, 22
    for i in range(4096):
        loss1.append(abs(y_predict[0][i] - y_real[0][i]) / y_real[0][i])
        list1.append(y_real[0][i])

    for i in range(4096):
        loss2.append(abs(y_predict[2][i] - y_real[2][i]) / y_real[2][i])
        list2.append(y_predict[2][i])

    for i in range(4096):
        predct1.append(y_predict[0][i])

    for i in range(4096):
        predct2.append(y_predict[2][i])

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.plot(list1, 'o-')
    ax1.plot(predct1, 'ro')
    ax2.plot(list2, 'o-')
    ax2.plot(predct2, 'ro')
    ax3.plot(loss1, 'o-')
    ax4.plot(loss2, 'o-')

    plt.plot()
    plt.show()
    '''
