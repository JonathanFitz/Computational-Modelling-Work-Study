# Train and test various models using nested cross-validation (cv)

# For each kind of feature creation method in FEATURES (sentiment, pronouns, etc),
# run the inner cv with all combination of hyperparameters listed in PARAM_GRID.
# Record the set of hyperparameters that give best average score in inner cv.

# Aftwards, choose the features/hyperparameter combo that gave best score on inner cv.
# Train this model on the training folds of the outer cv, and test on the test fold of the outer cv.

# Do this OUTER_FOLDS times, and for each combination of folds record the results in a text file.

# Note: When I talk about outer training folds (or, training folds of outer cv), I am referring to 
# the combination of outer folds used to train the model (at some point in time). Similiarly, 
# the outer test fold is the lone outer fold used to test the model (again, at some point in time). 
# THERE IS NO SEPERATE TRAINING AND TEST SET. The entire dataset is used in this nested cv, 
# and the outer test fold is never used to train the model. 


import numpy as np
import pandas as pd
from time import time, gmtime
from datetime import datetime
from sklearn.model_selection import KFold

import const
from header_files.rm_missing_scores import rm_missing_scores
from header_files.models import dense_net, conv_net, dense_bl, conv_bl, alt_bl, train_net, test_net

start_time = time()  # Time program execution


pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 500)

idx = pd.IndexSlice

# Import dataset containing all features, and give it some multilevel columns
dataset = pd.read_csv(const.DATASET, index_col=[1, 2, 3])
dataset = dataset.drop(labels='Unnamed: 0', axis=1)
dataset = dataset.sort_index(axis=0, level=0) 
dataset = dataset.astype(dtype='int16')

dataset = dataset.unstack().unstack().reorder_levels([2,1,0], axis=1)
dataset = dataset.sort_index(axis=1, level=[0, 1], sort_remaining=False)
dataset = dataset.dropna(axis=1, how='all') 

# Return dictionary of meetings where at least one participant has answered question
# Value of key is sum of scores for that meeting
meet_scores = rm_missing_scores()

# Response variable for each meeting are these scores
dataset['Response'] = [meet_scores[m] for m in dataset.index]

dataset = dataset.reset_index(drop=True)

#print(dataset.head())
#print(dataset.shape)


# Contains, for each combination of outer folds, the model chosen in inner cv and score on inner 
# and outer folds. Thus, dataframe will eventually have OUTER_FOLDS number of rows
outer_scores = pd.DataFrame(columns=['Model', 'Hyperparameters', 'Features', 
				     'Inner score', 'Outer score']) 

outer_num = 0  # Keep track of current outer fold 
 
# Create the folds for outer layer of nested cv
kf = KFold(n_splits=const.OUTER_FOLDS, shuffle=True, random_state=10)

for train, test in kf.split(dataset):

	outer_num = outer_num + 1

	# Holds best hyperparameters for each kind of feature
	inner_scores = pd.DataFrame(columns=[c for c in outer_scores.columns if c != 'Outer score']) 

	feat_num = 0  # Keep track of current feature

	for feature in const.FEATURES:

		feat_num = feat_num + 1

		# Select the data and features to be used in inner cv
		if feature != 'all':
			outer_train = dataset.loc[train, [feature, 'Response']]
			#print(outer_train.loc[:5, idx[:, :]])
		else:
			outer_train = dataset.loc[train, :]
			#print(outer_train.loc[:5, idx[:, :, :]])

		#print(outer_train.head())
		#print(outer_train.shape)

		# Start the inner cv, and return the best hyperparameters and score
		temp_inner = train_net(outer_train, start_time, outer_num, feat_num,
				       [c for c in inner_scores.columns if c != 'Features'])

		inner_scores = inner_scores.append(temp_inner, ignore_index=True)

		# For this best model, specify the method used to create features
		# This is done though only if the alternative baseline is not the best model,
		# since this model uses only the response variables (so, not the features)
		if inner_scores.loc[inner_scores.shape[0]-1, 'Model'] != 'Alternative Baseline':
			inner_scores.loc[inner_scores.shape[0]-1, 'Features'] = feature

	#print(inner_scores)

	# Select the hyperparameters/features that performed best in inner loops
	outer_scores = outer_scores.append(inner_scores.iloc[inner_scores['Inner score'].idxmin(), :], 
					   ignore_index=True)

	feat = outer_scores.loc[outer_scores.shape[0]-1, 'Features']  

	# Transform the outer folds to create chosen features. If feat is NaN, then 
	# alternative baseline was the best model and it doesn't matter which features are used. 
	if feat in [f for f in const.FEATURES if f != 'all']:

		outer_train = dataset.loc[train, [feat, 'Response']]
		outer_test = dataset.loc[test, [feat, 'Response']]
	else:
		outer_train = dataset.loc[train, :]
		outer_test = dataset.loc[test, :]

	model_type = outer_scores.loc[outer_scores.shape[0]-1, 'Model']
	hyperparams = outer_scores.loc[outer_scores.shape[0]-1, 'Hyperparameters']

	# Train model on all outer training folds and return score from outer test fold		
	score = test_net(outer_train, outer_test, model_type, hyperparams, feat, 
			 start_time, outer_num, feat_num)

	# Record this score
	outer_scores.loc[outer_scores.shape[0]-1, 'Outer score'] = score 

	#print(outer_scores)


# Save results in text file
results = open('results/{}_{}.txt'.format(const.NET_TYPE, datetime.today()), 'w')

results.write('This file contains the results of using nested cross-validation (cv)' +
              ' on the dataset. This consists of an outer and inner cv.\n\n' +

	      '\tNumber of outer cv folds: {}\n'.format(const.OUTER_FOLDS) +
	      '\tNumber of inner cv folds: {}\n\n'.format(const.INNER_FOLDS) +

	      'Dataset: {}\n'.format(const.TRANS_DIR) +
	      'Response question: {}\n\n'.format(const.RESPONSE))


results.write('\nARCHITECTURE AND HYPERPARAMETERS\n\n')
if const.NET_TYPE == 'dense':
	results.write('Architecture: {}\n'.format('Fully-connected (dense) net'))
elif const.NET_TYPE == 'conv':
	results.write('Architecture: {}\n'.format('Convolutional net'))

results.write('Batch size: {}\n'.format(const.BATCH_SIZE) +
	      'Max number of sentences examined in transcript: {}\n\n'.format(const.NUM_FEAT)) 

results.write('Feature derivation methods: {}\n\n'.format(const.FEATURES)) 
results.write('Possible hyperparameter values: {}\n\n'.format(const.PARAM_GRID)) 



# For each combination of outer folds, record various statistics
results.write('\nRESULTS\n\n')
for num in outer_scores.index:

	results.write('Fold combination #{}:\n'.format(num+1) +
		      '\tModel type: {}\n'.format(outer_scores.loc[num, 'Model']) +
		      '\tChosen hyperparameters: {}\n'.format(outer_scores.loc[num, 'Hyperparameters']) +
		      '\tChosen features: {}\n'.format(outer_scores.loc[num, 'Features']) +
		      '\tAverage inner cv score: {:.4f}\n'.format(outer_scores.loc[num, 'Inner score']) +
		      '\tOuter fold score: {:.4f}\n\n'.format(outer_scores.loc[num, 'Outer score']))

results.write('\nNOTES\n\n')
results.write('If the chosen model is a baseline model, then hyperparameters will be NaN since these are only relevant to neural nets.\n')
results.write('If the chosen model is the alternative baseline model, then features will also be NaN since the alt baseline only uses the response variable.\n\n')

time_elapsed = gmtime(time() - start_time)

results.write('Program running time: ' +
              '{} days, {} hours, {} minutes, {} seconds\n'.format(
              time_elapsed.tm_yday-1, time_elapsed.tm_hour, 
              time_elapsed.tm_min, time_elapsed.tm_sec))
results.close()
