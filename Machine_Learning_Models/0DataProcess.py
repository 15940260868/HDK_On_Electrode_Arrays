import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style ## 画的更好看
style.use('fivethirtyeight')

'''
## 输出格式：
pi = 3.1415926
print "pi: %10.3f" %pi
print '%010.3f' % pi  ## 用0填充空白
print('%-10.3f' % pi) ## 左对齐

## 遍历
i = 0
for i in range(len(data1)):
	print (i, data1[i])
	i += 1

print "***********"
for (i,v) in enumerate(data1):
	print (i,v)
'''

## ********************* 1. Numpy ********************
def basicNumpy():
	## np.array将系统自带的列表list转换成了numpy中的数组

	data1 = [1, 2, 3, 4, 5, 6]
	array1 = np.array(data1)
	# print("array1 is %s " % array1)

	## 嵌套列表会被转换为一个多维数组，它也可以被称为矩阵
	data2 = [[1, 2, 3], [4, 5, 6]]
	array2 = np.array(data2)
	# print( "array2 is:\n%s" %array2)

	## array数组需要注意的是，它内部的元素必须为相同类型，比如数值或者字符
	#print( "array2.dtype is %s" % array2.dtype)

	## ************************ 加减乘除 *****************************
	df1 = pd.DataFrame(np.arange(4).reshape(2, 2), columns=['a', 'b'])
	df2 = pd.DataFrame(np.arange(6).reshape(2, 3), columns=['a', 'b', 'c'])

	print(df2)
	print(df1.add(df2))
	print(df1.add(df2, fill_value = 10)) # 把NaN的值用 c的值 + fill_value

	## ************************ shape *****************************
	y = [1, 2, 3, 4]
	y = np.array(y)
	print("y.shape: {}".format(y.shape))
	print(y)

	y = y.reshape(-1, 1)  # 将行向量转化为列
	print("y.shape: {}".format(y.shape))
	print(y)


	# x1 = np.zeros((1,5))
	# print(x1.shape)
	# x2 = np.zeros(5)
	# x4 = np.zeros((5, 1))
	# x3 = np.zeros((3,4,5))
	#
	# print(x1)
	# print("*********")
	# print(x4)


def basicPandas():
	#  1. Series类似Numpy一维数组array ## 不同于array，都是竖着放，由一组数据和数据标签组成

	obj = pd.Series([1, 2, 3, 4])
	# print("obj[1]: {0}, \nobj.values: {1}, \nobj.index: {2}".format(obj[1], obj.values, obj.index));
	# print("obj[1]: %s, \nobj.values: %s, \nobj.index: %s" % (obj[1], obj.values, obj.index))


	s1 = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])
	# print s1[['a', 'd', 'e']] # series只能用方括号

	d = {'key1': 'wxy', "key2": "yyy"}
	s2 = pd.Series(d) # 默认按照字典的输出
	#s2 = pd.Series(d, index=['a', 'b', 'c'])
	#print(s2)  # 自定义的索引和字典队员不上时，会自动选择NaN

	'''
		2. DataFrame 既有行索引也有列索引, 每列可以存放不同数据类型
		读取行有三种方法，分别是loc，iloc，ix
		loc: 行标签索引来确定行的
		iloc: 行号
		ix: iloc和loc的集合
	'''
	dt = {"name": ['Bob', 'Alice', 'John'], "sex": ['Male', 'Female', 'Male'], "age": ['23', '22', '22']}
	df = pd.DataFrame(dt)
	# print df.age.astype
	# print "the first line: %s" % df.ix[0] ## 用索引字段ix获取一行
	# print "the first two lines: %s" % df[0:2] ## 用切片的形式获取行
	#print (df.age)
	# print df['name'] ## 方括号和.都可以获取，series只能用方括号

	df['Country'] = 'China'  # 列可以通过赋值的方式修改和添加，当列的名称是全新，则会在DataFrame的最右边自动加上新的一列
	df['age'] = [18, 19, 20]
	# print df[(df.sex == 'Male') & (df.age<20)]
	# print df.query(' sex == "Female" or name == "John" ') # query中可以直接使用列名

	df.index = ['a', 'b', 'c']
	#df.columns = ["aa", "bb", "cc", "dd"]
	print(df)
	print("*********************************")

	print(df.iloc[:-1, :-1])
	print("*********************************")

	X = df.iloc[:-1, :-1].values
	print(X, "\nX.shape: \n", X.shape)
	# print "*********"
	#print (df.iloc[1]) ## iloc 是行
	#print("*********************************")
	# print df.loc[1] ## loc是索引

	## 当行和列需要同时选择的时候，用逗号分割
	#print(df.ix['a':'c', 'age'])
	#print("*********************************")
	#print(df.ix[['a', 'c'], 'age'])


def readCol():
	d = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10,11,12], [13,14,15,16]]
	label = ["line0", "line1", "line2", "line3"]
	colLabel = ["col0", "col1", "col2", "col3"]
	df = pd.DataFrame(d, index=label, columns=colLabel)

	#print(df[0]) # 打印所有一维数组第一列
	#print(df.loc[:, 0])
	print("*****************")
	#print(df.iloc[:, 0]) # 印所有行第一列
	print(df.iloc[:,:]) # 所有行所有列

	print("*****************")
	print(df.iloc[:, :-1])
	#print(df.ix[:2, "col2"])
	#print(df.iloc[:2,:]) # 前两行
	#print("*****************") # 第二行开始
	#print(df.iloc[2:, 3])
	#print(df.ix[2:, ["col2", "col3"]])
	#print("*****************")
	#print(df.loc["line1",["col2", "col3"]])

	# print("Col0: \n".format(df[0]))

def basicOperations():
	df = pd.read_csv('s3.csv')
	df.columns = ['C1', 'C2','C3','C4','C5','C6','C7']
	c1=df['C1']
	c11 = c1.ix[df.index - 1]
	c11 = c11.reset_index(drop=True)
	print(len(df.index))

	df.plot()
	plt.legend()
	plt.show()

	df2 = pd.read_csv('s3.csv')
	print(len(df2.index))

	df.drop(df.index[len(df.index)])

	df2.drop(df2.index[len(df2.index)])

	print(df.tail(10))
	print("************")
	print(df2.tail(10))

	with open('m3.csv','ab') as f:
		f.write(open('s3.csv','rb').read())


	df3 = pd.read_csv('M2.csv')

	print(len(df3.index))
	print(df3.tail(10))


if __name__ == '__main__':
	basicNumpy()
	#basicPandas()
	#readCol()
