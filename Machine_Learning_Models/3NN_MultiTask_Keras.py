import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers
import datetime
import glob
from keras import backend

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

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

y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ") # (6400, 64)
y_train = np.reshape(y_train, (-1, 4096))
#scalarY = StandardScaler().fit(y_train)
#print("Y.mean: {}, std: {}".format(scalarY.mean_, scalarY.std_))
#y_std = scalarY.std_
#y_mean = scalarY.mean_
#y_train = scalarY.transform(y_train)


x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))
scalarX = StandardScaler().fit(x_train)
x_train = scalarX.transform(x_train)
#x_train = x_train[:150,:]

#x_train,x_test, y_train, y_test	 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0) # 372 93
#print(x_train.shape, x_test.shape) # (541, 4096) (136, 4096)

# (147, 136) (306, 156) (198, 198)
def initModel1(x_Input, out_name):
    out_model = Dense(4096, kernel_initializer='normal', activation='relu')(x_Input)
    out_model = Dense(1024, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(512, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(136, kernel_initializer='normal', name=out_name)(out_model)

    return out_model

def initModel2(x_Input, out_name):
    out_model = Dense(4096, kernel_initializer='normal', activation='relu')(x_Input)
    out_model = Dense(1024, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(512, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(156, kernel_initializer='normal', name=out_name)(out_model)

    return out_model
def initModel3(x_Input, out_name):
    out_model = Dense(4096, kernel_initializer='normal', activation='relu')(x_Input)
    out_model = Dense(1024, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(512, kernel_initializer='normal', activation='relu')(out_model)
    out_model = Dense(198, kernel_initializer='normal', name=out_name)(out_model)

    return out_model

def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def stackModel(x_first, y_first, x_second, y_second,x_last,y_last):
    x_Input1 = Input(shape=(4096,), name='input_1')
    x_Input2 = Input(shape=(4096,), name='input_2')
    x_Input3 = Input(shape=(4096,), name='input_3')

    model1 = initModel1(x_Input1, "model1")
    model2 = initModel2(x_Input2, "model2")
    model3 = initModel3(x_Input3, "model3")

    model = Model(inputs=[x_Input1, x_Input2, x_Input3], outputs=[model1, model2, model3])

    col = ["model1", "model2", "model3"]
    weights = [0.223, 0.472, 0.304]
    weightsDic = dict(zip(col, weights))

    loss = ['mean_squared_error']*3 # mean_squared_error
    lossDic = dict(zip(col, loss))

    model.compile(optimizer='adam',loss=lossDic,loss_weights = weightsDic,metrics=[rmse]) # metrics=['acc']

    y_train_col =[y_first, y_second, y_last]
    y_train_dict = dict(zip(col, y_train_col))
    # validation_data=(x_test, {"out1":y_test[:,0:1],"out2":y_test[:,1:2],"out3":y_test[:,2:3],"out4":y_test[:,3:4],"out5":y_test[:,4:5]})
    hist = model.fit({'input_1': x_first, 'input_2': x_second, 'input_3': x_last},y_train_dict , epochs=5, batch_size=10)

    model.save('Three_Model_UnS_Re.h5')
    kerasPlot(hist)


def kerasPlot(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot(loss_vec):
    plt.plot(loss_vec, 'r--', label='Test Loss')
    plt.title('Absolute Error')
    plt.legend(loc='upper right')
    plt.xlabel('Num Index')
    plt.ylabel('Absolute Error')
    plt.show()

loss_vec1 = []
loss_vec2 = []
loss_vec3 = []
def evaluteModel(x_test, y_test):
    # model = load_model('MLK_Model.h5')
    # model = load_model('MLK_Model.h5', custom_objects={'multitask_loss': multitask_loss})
    model = load_model('Three_Model_UnS_Re.h5',custom_objects = {'rmse': rmse})

    out1,out2,out3=  model.predict(x_test)
    print(out1.shape)
    # out1 *= y_std
    # out1 += y_mean
    #
    # out2 *= y_std
    # out2 += y_mean
    #
    # out3 *= y_std
    # out3 += y_mean

    y_test = y_test[0]
    print("out1.shape: {}, y_test.shape: {}".format(out1.shape, y_test.shape))
    # result = np.hstack((out1,out2,out3))
    # print("result.shape: {}, y_test.shape: {}".format(result.shape, y_test.shape))
    print("**********************")
    [rows, cols] = out1.shape
    cnt = 0

    for i in range(rows):
        for j in range(cols):
            loss1 = abs(out1[i][j]- y_test[i][j])
            loss2 = abs(out2[i][j] - y_test[i][j])
            loss3 = abs(out3[i][j] - y_test[i][j])
            loss_vec1.append(loss1)
            loss_vec2.append(loss2)
            loss_vec3.append(loss3)
            # if abs(loss) > 100000:
            #     cnt += 1
            #     print(result[i][j], y_test[i][j], loss)
            # else:
            #     loss_vec.append(loss)

    # print("cnt: {}".format(cnt))
    # plot(loss_vec1)
    # plot(loss_vec2)
    # plot(loss_vec3)
    print("loss_vec1.mean: {}, max: {}, min: {}".format(np.mean(loss_vec1), np.max(loss_vec1), np.min(loss_vec1)))
    print("loss_vec2.mean: {}, max: {}, min: {}".format(np.mean(loss_vec1), np.max(loss_vec2),np.min(loss_vec3)))
    print("loss_vec3.mean: {}, max: {}, min: {}".format(np.mean(loss_vec1),np.max(loss_vec2),np.min(loss_vec3)))


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # (147, 4096) (306, 4096) (198, 4096)
    x_first = loadtxtAndcsv_data("x_f.txt",',',np.float64)
    y_first = loadtxtAndcsv_data("y_f.txt",',',np.float64)
    print(x_first.shape, y_first.shape)

    x_second2 = loadtxtAndcsv_data("x_s.txt",',',np.float64)
    y_second2 = loadtxtAndcsv_data("y_s.txt",',',np.float64)
    print(x_second2.shape, y_second2.shape)
    x_second = x_second2[:147,:]
    y_second = y_second2[:147,:]

    x_last = loadtxtAndcsv_data("x_l.txt",',',np.float64)
    y_last = loadtxtAndcsv_data("y_l.txt",',',np.float64)
    print(x_last.shape, y_last.shape)
    x_last = x_last[:147,:]
    y_last = y_last[:147, :]

    #pcaY = PCA(n_components=5)
    # y_train = loadtxtAndcsv_data("y_train.txt", ",", np.float64)
    # y_test = loadtxtAndcsv_data("y_test.txt", ",", np.float64)
    # y_train = pcaY.fit_transform(y_train)
    # y_test = pcaY.transform(y_test)
    #y_test = pcaY.fit_transform(y_train)

    # pca = PCA(n_components=2)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)

    stackModel(x_first, y_first, x_second, y_second,x_last,y_last)
    # print("train time: {}".format((datetime.datetime.now() - starttime).seconds)) # 378S

    # x_test = generate_batches("../documents/test_output/", "/newOutput_64_*.txt", ",")
    # x_test = np.reshape(x_test, (-1, 4096))
    # scalar = StandardScaler().fit(x_test)
    #x_test = scalar.transform(x_test)
    x_test = x_second2[147:306,:]
    y_test = y_second2[147:306,:]

    # y_test = generate_batches("../documents/test_input/", "/newInput_*.txt", " ")
    # y_test = np.reshape(y_test, (-1, 4096))
    # print(np.mean(y_first),np.mean(y_second),np.mean(y_last), np.mean(y_test))
    print("x_test.shape: {}, y_test.shape :{}".format(x_test.shape, y_test.shape))
    # y_test = np.array(y_test)
    # y_test = scalar.transform(y_test)
    # print("x_test.shape: {}, y_test.shape: {}".format(x_test.shape, y_test.shape))

    x_test =[x_test] * 3
    y_test = [y_test] * 3
    evaluteModel(x_test, y_test)

    #print("mean: {0}, max: {1}, min: {2}".format(np.mean(loss_vec), np.max(loss_vec), np.min(loss_vec)))
    print("final time: {}".format((datetime.datetime.now() - starttime).seconds))