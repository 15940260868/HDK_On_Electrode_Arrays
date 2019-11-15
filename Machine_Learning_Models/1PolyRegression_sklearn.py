import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# seed
rng = np.random.RandomState(123)

# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
	return np.loadtxt(fileName, delimiter=split, dtype=dataType)

# construct samples. give a x, generate y with noise
def genY(x):
	a0, a1, a2, a3, e = 0.1, -0.02, 0.03, -0.04, 0.05
	yr = a0 + a1 * x + a2 * (x ** 2) + a3 * (x ** 3) + e
	y = yr + 0.03 * rng.rand(1)
	return y


# plot
plt.figure()
plt.title('polynomial regression(sklearn)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

x_tup = np.linspace(-1, 1, 30)
y = [genY(a) for a in x_tup]
print(y)
x = x_tup.reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
plt.plot(x, y, 'k.')

qf = PolynomialFeatures(degree=3)
qModel = LinearRegression()
qModel.fit(qf.fit_transform(x), y)
print('----')
print(qf.get_params())

xp = np.linspace(-1, 2, 100)
yp = qModel.predict(qf.transform(xp.reshape(-1, 1)))

plt.plot(xp, yp, 'r-')
plt.show()

