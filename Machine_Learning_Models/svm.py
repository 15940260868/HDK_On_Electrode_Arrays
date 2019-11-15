import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 3 + 0.1 * x1
    return y


def load_data():
    x1_train = np.linspace(0,50,500)
    x2_train = np.linspace(-10,10,500)
    data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
    x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
    return data_train, data_test

train, test = load_data()
x_train, y_train = train[:,:2], train[:,2] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
x_test ,y_test = test[:,:2], test[:,2] # 同上,不过这里的y没有噪声
SVM_loss = []

def try_different_method(model, lossList):
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    print(y_predict)

    score = model.score(x_test, y_test)
    print("{0}.score: {1}".format(model, score))

    for i in range(len(y_predict)):
        loss = abs(y_predict[i] - y_test[i])
        lossList.append(loss)

    return lossList

def SVM():
    SVM_Model = svm.SVR(kernel='linear') # kernel='linear' # kernel='poly' # kernel='rbf'
    try_different_method(SVM_Model, SVM_loss)
    print("SVM:\nmean: {}, max: {}, min: {}".format(np.mean(SVM_loss), np.max(SVM_loss), np.min(SVM_loss)))

if __name__ == '__main__':
    SVM()