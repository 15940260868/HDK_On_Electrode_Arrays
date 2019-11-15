# https://blog.csdn.net/juyin2015/article/details/78679707
# https://blog.csdn.net/u013555719/article/details/79300745

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_classes = 4096  # 输出大小
input_size = 4096  # 输入大小
hidden_units_size = 64  # 隐藏层节点数量
batch_size = 100
training_iterations = 100


def loadtxtAndcsv_data(fileName, split, dataType):
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def fully_connected(input_layer, weights, biases):
	layer = tf.add(tf.matmul(input_layer, weights), biases)
	return (tf.nn.tanh(layer))


def init_weight(shape, st_dev):
	weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
	return (weight)

'''
def init_bias(shape, st_dev):
	bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
	return (bias)
'''

def init_bias(value, shape):
	bias = tf.Variable(tf.constant(value, shape))
	#bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
	return (bias)

loss_vec = []
test_loss = []
def train():
	dataX = loadtxtAndcsv_data("../documents/input.txt", "\t", np.float64)
	dataY = loadtxtAndcsv_data("../documents/output.txt", ",", np.float64)

	y_train = generate_batches("../documents/MLK_Input/", "/newInput_*.txt", " ")  # (6400, 64)
	y_train = np.reshape(y_train, (-1, 4096))  # 100 * 4096
	# y_train = np.argmax(y_train, axis=1)

	x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
	x_train = np.reshape(x_train, (-1, 4096))  # 100 * 4096
	# x_train = np.argmax(x_train, axis=1)

	# x_test = loadtxtAndcsv_data("../documents/MLK_Output/newOutput_64_100.txt", ",", np.float64).reshape(-1, 4096)
	# y_test = loadtxtAndcsv_data("../documents/MLK_Input/newInput_100.txt", " ", np.float64).reshape(-1, 4096)

	x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)  # 372 93
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	print("max: {}, min: {}".format(np.max(y_train), np.min(y_train)))

	x_data = np.reshape(np.array(dataX[:, :], dtype=np.float64), (1, -1))
	y_data = np.reshape(np.array(dataY[:, :], dtype=np.float64), (1, -1))

	# 定义placeholder
	X = tf.placeholder(tf.float32, shape=[None, input_size])
	Y = tf.placeholder(tf.float32, shape=[None, num_classes])

	# ***************** 神经网络第一层 64个节点 *****************
	weight_1 = init_weight(shape=[input_size, hidden_units_size], st_dev=0.1)
	biases_1 = tf.Variable(tf.constant(0.1), [64])
	layer_1 = fully_connected(X, weight_1, biases_1)

	# *****************  第二层, 32 个节点 *******************
	weight_2 = init_weight(shape=[hidden_units_size, 32], st_dev=0.1)
	biases_2 = tf.Variable(tf.constant(0.1), [32])
	layer_2 = fully_connected(layer_1, weight_2, biases_2)

	# *****************  第三层, 16 个节点 *******************
	weight_3 = init_weight(shape=[32, 16], st_dev=0.1)
	biases_3 = tf.Variable(tf.constant(0.1), [16])
	layer_3 = fully_connected(layer_2, weight_3, biases_3)

	# ***************** 神经网络输出层*******************
	weight_4 = init_weight(shape=[16, num_classes], st_dev=0.1)
	biases_4 = tf.Variable(tf.constant(0.1), [num_classes])
	output = tf.matmul(layer_3, weight_4) + biases_4

	# 损失函数
	loss = tf.reduce_mean(tf.square(Y - output))
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))


	# 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss) # AdamOptimizer GradientDescentOptimizer

	# 初始化变量
	init = tf.global_variables_initializer()


	with tf.Session() as sess:
		# 变量初始化
		sess.run(tf.global_variables_initializer())
		for i in range(training_iterations):
			# sess.run([train_step, loss], feed_dict={X: x_data, Y: y_data})
			sess.run(train_step, feed_dict={X: x_data, Y: y_data})

			temp_loss = sess.run(loss, feed_dict={X: x_data, Y: y_data})
			loss_vec.append(temp_loss)  # 将训练集上的误差存进loss_vec中

			if (i + 1) % 10 == 0:
				print('Generation: ' + str(i + 1) + ', Loss = ' + str(temp_loss))

			#test_temp_loss = sess.run(loss, feed_dict={X: x_data, Y: y_data})
			#test_loss.append(test_temp_loss)  # 将测试集上的误差存进test_loss中

			# 获得预测值
		prediction_value = sess.run(output, feed_dict={X: x_data})


		prediction_value = np.reshape(prediction_value, (64, 64))
		print("******************************")
		print(prediction_value)





if __name__ == '__main__':
	train()