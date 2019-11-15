from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import ShuffleSplit
import glob
import matplotlib.pyplot as plt
import numpy as np
import visuals as vs

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
y_train = np.reshape(y_train, (-1, 4096)) # 100 * 4096
# y_train = np.argmax(y_train, axis=1)

x_train = generate_batches("../documents/MLK_Output/", "/newOutput_64_*.txt", ",")
x_train = np.reshape(x_train, (-1, 4096))  # 100 * 4096

x_test = loadtxtAndcsv_data("../documents/MLK_Output/newOutput_64_100.txt", ",", np.float64).reshape(-1, 4096) # (64, 64)
y_test = loadtxtAndcsv_data("../documents/MLK_Input/newInput_100.txt", " ", np.float64).reshape(-1, 4096)

def performance_metric(y_true, y_predict):
	score = r2_score(y_true, y_predict)
	return score

def fit_model(X, y):
	""" Performs grid search over the 'max_depth' parameter for a
		decision tree regressor trained on the input data [X, y]. """
	# Create cross-validation sets from the training data
	cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
	# cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)
	# TODO: Create a decision tree regressor object
	regressor = DecisionTreeRegressor()
	# TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
	params ={'max_depth':[3,5,1]}
	# TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
	scoring_fnc = make_scorer(performance_metric, greater_is_better=True)

	# TODO: Create the grid search object
	grid = GridSearchCV(estimator = regressor, param_grid = params, scoring = scoring_fnc, cv = cv_sets)
	# grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator)
	# Fit the grid search object to the data to compute the optimal model
	grid = grid.fit(X, y)
	# Return the optimal model after fitting the data
	return grid.best_estimator_


predict_list = []
real_list = []
loss_list = []
if __name__ == '__main__':
	# score = performance_metric([69463.426476,80466.511639,93890.510385], [69205.637356,79265.518928,94458.660875])
	# print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

	# Fit the training data to the model using grid search
	reg = fit_model(x_train, y_train)

	# 根据不同的训练集大小，和最大深度，生成学习曲线
	#vs.ModelLearning(x_train, y_train)
	# 根据不同的最大深度参数，生成复杂度曲线
	#vs.ModelComplexity(x_train, y_train)

	# Produce the value for 'max_depth'
	print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

	# display(reg.get_params()['max_depth'])
	predicted_test = reg.predict(x_test)
	# print(predicted_test)
	# print(predicted_test.shape)
	for i in range(len(predicted_test)):
		for j in range(len(predicted_test[0])):
			predict_list.append(predicted_test[i][j])
			real_list.append(y_test[i][j])
			loss = abs(predicted_test[i][j] - y_test[i][j])
			loss_list.append(loss)

	print("loss_list:\nmean: {}, max: {}, min: {}".format(np.mean(loss_list), np.max(loss_list),np.min(loss_list)))
	test_set_scoring = performance_metric(predict_list, real_list)
	print('The test set score is {}'.format(test_set_scoring))