import numpy as np
np.set_printoptions(suppress=True)
import pandas
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Input, Embedding, LSTM, Dense
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

import glob

loss_vec = []

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
y_train = np.reshape(y_train, (-1, 4096)) # 100 * 4096
# y_train = np.argmax(y_train, axis=1)

x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))  # 100 * 4096

x_train,x_test, y_train, y_test	 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0) # 372 93

x_train -= np.mean(x_train, axis=0)  # 减去均值，使得以0为中心
x_train /= np.std(x_train, axis=0)  # 归一化

x_test -= np.mean(x_test, axis=0)  # 减去均值，使得以0为中心
x_test /= np.std(x_test, axis=0)  # 归一化

def loadtxtAndcsv_data(fileName, split, dataType):
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4096*2, input_dim=4096, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4096*2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4096*2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4096, kernel_initializer='normal'))

	# Compile model, 选定loss函数和优化器
	#sgd = optimizers.SGD(lr=0.01, clipvalue=0.5) #  clipnorm=1.
	#adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def kerasPlot(history):
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


# https://blog.csdn.net/u012193416/article/details/79376345
def train(x_train):
	print('Training -----------')
	y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ")
	y_train = np.reshape(y_train, (-1, 4096)) # 100

	x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
	x_train = np.reshape(x_train, (-1, 4096))  # 63

	y_test = loadtxtAndcsv_data("../documents/MLK_Input/newInput_60.txt", " ", np.float64)
	y_test = np.reshape(np.array(y_test[:, :], dtype=np.float64), (1, -1))

	x_test = loadtxtAndcsv_data("../documents/MLK_Output/newOutput_64_60.txt", ",", np.float64)
	x_test = np.reshape(np.array(x_test[:, :], dtype=np.float64), (1, -1))

	model = baseline_model()
	hist = model.fit(x_train, y_train[:,:], nb_epoch=100, batch_size=10)

	'''
	for step in range(100):
		loss = model.train_on_batch(x_train[:63,:], y_train)
		loss = hist.history
		loss_vec.append(loss)
		if step % 10 == 0:
			print("Generation: %d, the loss: %f" % (step, loss))
	'''
	# print(hist.history.keys())
	kerasPlot(hist)

	# Y_pred = model.predict(x_test)
	# Y_pred = np.reshape(Y_pred, (64, 64))
	# print('test before save: ', Y_pred)

	model.save('MLK_Model.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
	del model  # deletes the existing model

def plot():
	#plt.plot(loss_vec, 'r-', label='Train Loss')
	plt.plot(loss_vec, 'r--', label='Test Loss')
	# plt.title('Loss (MSE) per Generation')
	# plt.legend(loc='upper right')
	#plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()

# load
def evaluteModel(x_test):
	model = load_model('MLK_Model.h5')
	# x_test = loadtxtAndcsv_data("../documents/MLK_Input/newInput_60.txt", " ", np.float64)
	# x_test = np.reshape(np.array(x_test[:, :], dtype=np.float64), (1, -1))
	#
	# y_real = loadtxtAndcsv_data("../documents/MLK_Output/newOutput_64_60.txt", ",", np.float64)
	# y_real = np.reshape(np.array(y_real[:, :], dtype=np.float64), (1, -1))

	# x_test = pca.transform(x_test)
	y_predict =  model.predict(x_test)
	print('predicted y: ',y_predict)
	#print("**********************")
	#print("real y: {}".format(x_test))
	print("**********************")

	for i in range(len(y_predict)):
		for j in range(len(y_predict[0])):
			loss = abs(y_predict[i][j] - y_test[i][j])
			loss_vec.append(loss)

	print("mean: {0}, max: {1}, min: {2}".format(np.mean(loss_vec),np.max(loss_vec), np.min(loss_vec)))
	plot()



if __name__ == '__main__':
	train(x_train)
	evaluteModel()