# Defines various models

import numpy as np
import pandas as pd
from time import time, gmtime

#import tensorflow as tf

from keras import backend as K, layers
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import normalize

import const


#current_model = 0  # keep track of current model

# Creates structure of fully-connected (dense) neural network
def dense_net(num_feat, outer_num, feat_num, start_time, num_layer=2, neurons_per_layer=500, 
	      perc_dropout=0, optimizer='rmsprop', learning_rate=1e-4, act_func='relu', epochs=100):

	K.clear_session()

	#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
	
	#global current_model 
	#current_model = current_model + 1
	#print('Currently fitting model {}\{}'.format(current_model, num_models))

	print('Current outer fold: {}\{}'.format(outer_num, const.OUTER_FOLDS))
	print('Current feature: {}\{}'.format(feat_num, len(const.FEATURES)))
	
	time_elapsed = gmtime(time() - start_time)
	print('Program running time: ' +
	      '{} days, {} hours, {} minutes, {} seconds\n'.format(
	      time_elapsed.tm_yday-1, time_elapsed.tm_hour, 
	      time_elapsed.tm_min, time_elapsed.tm_sec))
	
	# Define neural network architecture
	model = Sequential()
	model.add(Dense(neurons_per_layer, activation=act_func,
	                input_shape=(num_feat, )))

	for layer in range(num_layer-2):
		model.add(Dense(neurons_per_layer, activation=act_func))
		model.add(Dropout(perc_dropout))

	model.add(Dense(neurons_per_layer, activation=act_func))
	model.add(Dense(1))
	
	#model.summary()
	
	#model.compile(optimizer, loss='mse', metrics=['mae'], options = run_opts)
	model.compile(optimizer, loss='mse', metrics=['mae'])

	return model


# Creates structure of convolutional neural network
def conv_net(num_feat, outer_num, feat_num, start_time, num_layer=2, filters=500, kernel_size=5,
             strides=1, padding='valid', act_func='relu', pool_type = 'MaxPooling1D', 
	     pool_size=2, pool_strides=None, pool_padding='valid',
             neurons_per_layer=100, optimizer='rmsprop', epochs=100):

	pool_dict = {'MaxPooling1D': layers.MaxPooling1D, 
		     'AveragePooling1D': layers.AveragePooling1D,
		     'GlobalMaxPooling1D': layers.GlobalMaxPooling1D, 
		     'GlobalAveragePooling1D': layers.GlobalAveragePooling1D
		    }

	K.clear_session()

	#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

	#global current_model
	#current_model = current_model + 1
	#print('Currently fitting model {}\{}'.format(current_model, num_models))

	print('Current outer fold: {}\{}'.format(outer_num, const.OUTER_FOLDS))
	print('Current feature: {}\{}'.format(feat_num, len(const.FEATURES)))
	
	time_elapsed = gmtime(time() - start_time)
	print('Program running time: ' +
	      '{} days, {} hours, {} minutes, {} seconds\n'.format(
	      time_elapsed.tm_yday-1, time_elapsed.tm_hour,
	      time_elapsed.tm_min, time_elapsed.tm_sec))
	
	# Define neural network architecture
	model = Sequential()
	model.add(layers.Conv1D(filters, kernel_size, strides=strides,
	                        padding=padding, activation=act_func,
	                        input_shape=(num_feat, 1)))
	
	if pool_type in ('MaxPooling1D', 'AveragePooling1D'):
		model.add(pool_dict[pool_type](pool_size=pool_size, strides=pool_strides,
					       padding=pool_padding))
	elif pool_type in ('GlobalMaxPooling1D', 'GlobalAveragePooling1D'):
		model.add(pool_dict[pool_type]())
		model.add(layers.Reshape((filters, 1)))
		
	for layer in range(num_layer-2):
		model.add(layers.Conv1D(filters, kernel_size, strides=strides,
			  		padding=padding, activation=act_func))
	
		if pool_type in ('MaxPooling1D', 'AveragePooling1D'):
			model.add(pool_dict[pool_type](pool_size=pool_size, strides=pool_strides,
						       padding=pool_padding))
		elif pool_type in ('GlobalMaxPooling1D', 'GlobalAveragePooling1D'):
			model.add(pool_dict[pool_type]())
			model.add(layers.Reshape((filters, 1)))
	
	model.add(layers.Flatten())
	
	model.add(layers.Dense(neurons_per_layer, activation=act_func))
	model.add(layers.Dense(1))
	#model.summary()

	#model.compile(optimizer, loss='mse', metrics=['mae'], options = run_opts)
	model.compile(optimizer, loss='mse', metrics=['mae'])
	return model


# First baseline model for dense neural net
# This model was provided by Dr. Murray
def dense_bl(X_train, y_train, X_test, y_test):

	num_feat = X_train.shape[1]

	# Define neural network architecture
	model = Sequential()
	model.add(layers.Dense(500, activation='relu', input_shape=(num_feat, )))
	model.add(layers.Dense(500, activation='relu'))
	model.add(layers.Dense(1))

	model.compile(optimizer=RMSprop(lr=1e-4), loss='mse', metrics=['mae'])

	model.fit(x=X_train, y=y_train, epochs=100, verbose=0)
	 
	# Record test mse and mae
	test_MSE, test_MAE = model.evaluate(x=X_test, y=y_test, verbose=0)

	#print('test_MSE for dense baseline: {}\n'.format(test_MSE))
	#print('test_MAE for dense baseline: {}\n'.format(test_MAE))

	return test_MSE 


# First baseline model for convolutional neural net
# This model was provided by Dr. Murray
def conv_bl(X_train, y_train, X_test, y_test):

	# Add a dimension to inputs
	X_train = np.expand_dims(X_train, axis=2)  
	X_test = np.expand_dims(X_test, axis=2)  
	
	num_feat = X_train.shape[1]
	
	# Define neural network architecture
	model = Sequential()
	model.add(layers.Conv1D(500, 5, activation='relu', input_shape=(num_feat, 1)))
	model.add(layers.GlobalMaxPool1D())
	model.add(layers.Dense(100, activation='relu'))
	model.add(layers.Dense(1))
	 
	model.compile(optimizer=RMSprop(lr=1e-4), loss='mse', metrics=['mae'])

	model.fit(x=X_train, y=y_train, epochs=200, verbose=0)
	 
	# Record test mse and mae
	test_MSE, test_MAE = model.evaluate(x=X_test, y=y_test, verbose=0)

	return test_MSE 


# Alternative baseline model
# X_train, X_test just here for compatability
def alt_bl(X_train, y_train, X_test, y_test):

	# This model uses the average response from the training set to predict the 
	# response on test set

	test_pred = [y_train.mean()]*len(y_test)
	test_MSE = mean_squared_error(test_pred, y_test)
	#test_MAE = mean_absolute_error(test_pred, y_test)
		
	return test_MSE


# Performs cross-validation with the baseline models
def cross_val(data_set, cols):

	ave_scores = []

	# Create two baseline models: One corresponding to either dense or 
	# convolutional net, and then alternative baseline
	for bl in [const.NET_TYPE, 'alt']:

		scores = []	

		cv = KFold(n_splits=const.INNER_FOLDS, shuffle=True, random_state=15)

		for train, test in cv.split(data_set):

			X_train = normalize(data_set.iloc[train, :-1])
			y_train = data_set.iloc[train, -1].values

			X_test = normalize(data_set.iloc[test, :-1])
			y_test = data_set.iloc[test, -1].values

			test_score = globals()[bl+'_bl'](X_train, y_train, X_test, y_test)	
 
			scores.append(test_score)
		
		ave_scores.append(sum(scores)/const.INNER_FOLDS)

	return pd.DataFrame([['Default Baseline', float('nan'), ave_scores[0]], 
			     ['Alternative Baseline', float('nan'), ave_scores[1]]], columns=cols)


# Handles the inner loop of the nested cross-validation, and returns, for a particular set of features,
# the best hyperparameters in param_grid and corresponding average cv score
def train_net(data_set, start_time, outer_num, feat_num, cols):

	#global current_model
	#current_model = 0
		
	model_info = pd.DataFrame(columns=cols)  
 
	X = normalize(data_set.drop('Response', axis=1))
	y = data_set['Response'].values
 
	## Find number of models to be built
	#num_models = 1
	#for value in const.PARAM_GRID.values():
	#	num_models = num_models*len(value)
	#num_models = num_models*const.INNER_FOLDS

	# Run baseline models
	model_info = model_info.append(cross_val(data_set, cols), ignore_index=True)

	# Build neural net
	model = KerasRegressor(build_fn=globals()[const.NET_TYPE+'_net'], num_feat=X.shape[1], 
			       outer_num=outer_num, feat_num=feat_num, start_time=start_time)

	grid = GridSearchCV(model, const.PARAM_GRID, scoring='neg_mean_squared_error',
	                    cv=const.INNER_FOLDS, verbose=1)

	# For convolutional network, add a dimension to inputs
	if const.NET_TYPE == 'conv':
	    	X = np.expand_dims(X, axis=2)  

	grid.fit(X, y, batch_size=const.BATCH_SIZE, verbose=1)

	model_info = model_info.append(pd.DataFrame([['Neural Net', grid.best_params_, 
				       -grid.best_score_]], columns=cols), ignore_index=True)

	#print('Best params are: {}\n'.format(grid.best_params_))

	#print(model_info)

	# Return the model that has the best (lowest) average cv score
	return model_info.iloc[model_info['Inner score'].idxmin(), :]


# Train and test a model on provided data, features, and hyperparameters.
# Note that train_net chose these features and hyperparameters  
def test_net(train_set, test_set, model_type, hyperparams, feat, start_time, outer_num, feat_num):

	#global current_model
	#current_model = 0

	#num_models = 1

	X_train = normalize(train_set.drop('Response', axis=1))
	y_train = train_set['Response'].values

	X_test = normalize(test_set.drop('Response', axis=1))
	y_test = test_set['Response'].values

	if model_type == 'Default Baseline':
	
		test_MSE = globals()[const.NET_TYPE+'_bl'](X_train, y_train, X_test, y_test)
	
	elif model_type == 'Alternative Baseline':
	
		test_MSE = alt_bl(X_train, y_train, X_test, y_test)

	elif model_type == 'Neural Net':

		# For convolutional network, add another dimension
		if const.NET_TYPE == 'conv':
		    	X_train = np.expand_dims(X_train, axis=2)  
		    	X_test = np.expand_dims(X_test, axis=2)  

		model = globals()[const.NET_TYPE+'_net'](num_feat=X_train.shape[1], outer_num=outer_num,
				  feat_num=feat_num, start_time=start_time, **hyperparams)

		model.fit(x=X_train, y=y_train, batch_size=const.BATCH_SIZE, 
			  epochs=hyperparams['epochs'])

		# Record test mse and mae
		test_MSE, test_MAE = model.evaluate(x=X_test, y=y_test, batch_size=const.BATCH_SIZE, 
						    verbose=1)

	#print('\nModel used is {}'.format(model_type))
	#print('Metrics are: {}\n'.format(model.metrics_names))
	#print('Test score is {}\n'.format(test_score[0]))
	
	return test_MSE 
