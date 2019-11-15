# -*- coding:utf-8 -*-
import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Conv2D,MaxPool2D,Dense,Flatten,MaxPooling2D
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import datetime
import glob

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

# (137, 128) # #(308,256) # (197,164) # (100, 84)

#(308,256)
def baseline_model2():
    model = Sequential()
    model.add(Dense(512*4, input_dim=512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512*4, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(512*4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(64, 64), strides=(1, 1),activation='relu',input_shape=(4096,4096,0)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(512, kernel_initializer='normal'))
    # model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model
    # model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])


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
    model = baseline_model()

    hist = model.fit(x_train, y_train, epochs=100, batch_size=100)
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
    y_real = loadtxtAndcsv_data("y512_real.txt", ",", np.float64)
    # y_real = np.reshape(y_real, (-1, 4096))
    y_test = y_real

    '''
    pca = PCA(n_components=512)
    y_real = pca.fit_transform(y_real)
    # y_real = model.transform(y_real) # # 标准化 336
    print("y_real_scala_reduced: {}".format(y_real.shape))
    print(y_real[0:1,0:5])
    '''
    model = load_model('ec2_512.h5')
    y_predict =  model.predict(x_test) # # 预测出来的被标准化的512
    print("y_predict: {}".format(y_predict.shape)) 
    print(y_predict[0:1,0:5])
    # y_predict = pca.inverse_transform(y_predict) # 预测出来的512还原到4096
    print("y_predict_recover: {}".format(y_predict.shape))

    [rows, cols] = y_test.shape

    for i in range(rows):
        for j in range(cols):
            loss = abs(y_predict[i][j] - y_test[i][j]) / y_test[i][j]
            loss_vec.append(loss)

    print("loss_vec.mean: {}, max: {}, min: {}".format(np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))
    return loss_vec

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
  # print("feature.shape: {}".format(feature.shape))
  #get new data
  data=np.dot(norm_X,np.transpose(feature)) # (247, 4096)
  print(data.shape)
  print('\nOriginal Method:\n')
  print(data[1:5, :2].real)
  return data.real

if __name__ == '__main__':
    starttime = datetime. datetime.now()

    # x_train = loadtxtAndcsv_data("x_false_512.txt", ",", np.float64)
    # y_train = loadtxtAndcsv_data("y_false_512.txt", ",", np.float64)

    # x_real_reduced = loadtxtAndcsv_data("x_real_512.txt", ",", np.float64)
	# print(x_train.shape, y_train.shape,x_real_reduced.shape) # (18900, 378) (18900, 378) (378, 378) (378, 378)
    y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ")  # 677
    y_train = np.reshape(y_train, (-1, 4096))

    x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
    x_train = np.reshape(x_train, (-1, 4096))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    print(x_train.shape, y_train.shape)
    train(x_train, y_train)

    ##################################################################################
    #                                  Evaluation
    ##################################################################################
    loss = evaluteModel(x_test, loss_vec) # x_real_reduced
    loss_arr = np.array(loss)
    loss_arr = np.reshape(loss_arr, (-1, 4096))
    np.savetxt("ec2_loss_arr_512.txt", loss_arr, fmt="%.6f", delimiter=",")

    loss_arr2 = loadtxtAndcsv_data("ec2_loss_arr_512.txt", ",", np.float64)
    print("loss_arr_512.shape :{}, mean: {}, max: {}, min: {}".format(loss_arr2.shape, np.mean(loss_arr2),np.max(loss_arr2), np.min(loss_arr2)))


    dict = {'row': []}
    [row, col] = loss_arr2.shape
    for i in range(row):
        cnt = 0
        for j in range(col):
            if (loss_arr2[i][j] > 0.05):
                cnt += 1
        dict["row"].append(cnt)

    value, = dict.values()
    ratioCnt = 0
    for i in range(len(value)):
        if value[i] != 0:
            # print(i, value[i])
            ratioCnt += 1
    print("ratioCnt:{} ".format(ratioCnt))