# https://www.jianshu.com/p/a6cf116c1bde
import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
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


y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ") # 677
y_train = np.reshape(y_train, (-1, 4096))
model = PCA(n_components=32).fit(y_train)  # (677, 4096)
y_train = model.transform(y_train)

x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))
scalarX = StandardScaler().fit(x_train)
x_train = scalarX.transform(x_train)
model2 = PCA(n_components=32).fit(x_train)  # (677, 4096)
x_train = model2.transform(x_train)

print("x_train.shape: {}, y_train.shape: {}".format(x_train.shape, y_train.shape))

# x_test = generate_batches("../documents/test_output/", "/newOutput_64_*.txt", ",")
# x_test = np.reshape(x_test, (-1, 4096))
# scalar = StandardScaler().fit(x_test)
# x_test = scalar.transform(x_test)
# model2 = PCA(n_components=262).fit(x_test)
# x_test = model2.transform(x_test)
#
# y_test = generate_batches("../documents/test_input/", "/newInput_*.txt", " ")
# y_test = np.reshape(y_test, (-1, 4096))
# y_test = model2.transform(y_test)
x_train,x_test, y_train, y_test	 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
print("x_test.shape: {}, y_test.shape: {}".format(x_test.shape, y_test.shape))


def baseline_model():
    model = Sequential()
    model.add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal'))

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
    model = baseline_model()

    hist = model.fit(x_train, y_train, epochs=100, batch_size=5)
    # print(hist.history.keys())
    kerasPlot(hist)
    model.save('OneModel_UnS_262.h5' )  # HDF5 file, you have to pip3 install h5py if don't have it
    del model  # deletes the existing model

def plot():
    plt.plot(loss_vec, 'r--', label='Test Loss')
    plt.ylabel('Loss')
    plt.show()

# load
def evaluteModel(x_test, y_test):
    model = load_model('OneModel_UnS_262.h5')
    y_predict =  model.predict(x_test)

    print("x_test: {}, y_predict: {}, y_test: {}".format(x_test.shape, y_predict.shape, y_test.shape))
    [rows, cols] = y_predict.shape
    loss_vec = []
    for i in range(rows):
        for j in range(cols):
            loss = abs(y_predict[i][j] - y_test[i][j])
            loss_vec.append(loss)

    print("loss_vec.mean: {}, max: {}, min: {}".format(np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))

if __name__ == '__main__':
    starttime = datetime. datetime.now()
    # x_train -= np.mean(x_train, axis=0)  # 减去均值，使得以0为中心
    # x_train /= np.std(x_train, axis=0)  # 标准化

    train(x_train,  y_train)
    evaluteModel(x_test, y_test)
    print("final time: {}".format((datetime.datetime.now() - starttime).seconds))
