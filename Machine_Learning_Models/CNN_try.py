# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import glob
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model



# 数据处理
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

'''
y_train = generate_batches("../documents/900_input/", "/newInput_*.txt", " ")
print(y_train.shape) # (1280, 64)
y_train = y_train.reshape(20, 4096)

x_train = generate_batches("../documents/900_output/", "/newOutput_64_*.txt", ",")
x_train = x_train.reshape(20,64,64,1)

print(x_train.shape, y_train.shape) # (20, 64, 64, 1) (20, 4096)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=666)

'''
x_train = loadtxtAndcsv_data("x_false_4096.txt", ",", np.float64)
x_train = x_train.reshape(1400,64,64,1)

y_train = loadtxtAndcsv_data("y_false_4096.txt", ",", np.float64)
y_train = y_train.reshape(1400,4096)

x_test = loadtxtAndcsv_data("x_real_4096.txt", ",", np.float64)
x_test = x_test.reshape(600, 64,64,1)

y_test = loadtxtAndcsv_data("y4096_real.txt", ",", np.float64)
y_test = y_test.reshape(600,4096)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (64,64,1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten()) # 拉成一维数据
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.2))
    # model.add(Dense(10, activation = "softmax"))
    model.add(Dense(4096, kernel_initializer='normal')) # 全连接层

    # model.compile(optimizer = "SGD" , loss = "categorical_crossentropy", metrics=["accuracy"])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    return model


def train():
    model = build_model()
    model.fit(x_train, y_train, batch_size=100, verbose=1,epochs=5,validation_data=(x_test, y_test))
    print(model.summary())

    model.save('my_model.h5')
    del model



loss_vec= []
def evaluteModel( ):
    model = load_model('my_model.h5')
    y_predict = model.predict(x_test)

    [rows, cols] = y_test.shape

    for i in range(rows):
        for j in range(cols):
            loss = abs(abs(y_predict[i][j]) - abs(y_test[i][j])) / abs(y_test[i][j])
            loss_vec.append(loss)


    print("loss_vec.mean: {}, max: {}, min: {}".format(np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))
    return loss_vec

if __name__ == '__main__':
    train()
    evaluteModel()
