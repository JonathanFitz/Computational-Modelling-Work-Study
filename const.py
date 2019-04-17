# This file holds various values that the user can alter to change program functionality.
# These would normally be set to constants in a language that supported constants.



# DATASET CREATION

# Directory containing original transcripts
# The names of the transcripts are important. Everything before the first period (ex. ES2002a)
# is used to uniquely identify each transcript
TRANS_DIR = 'data/original_dataset/'

# Directory containing all feature data
# The names of these directories are important. They are used to uniquely identify the kind
# of feature creation technique
FEATS_DIR = 'data/features/'

DATASET = 'data/dataset.txt'  # Dataset containing all features generated from transcripts
MEET_SCORES = 'data/meeting_reviews.csv'  # Various questionnaire responses for each meeting

RESPONSE = 'vraag16'  # Questionnaire answer to be used as response variable
NUM_FEAT = 1500  # Max number of sentences scored in a transcript
		 # Meetings with less than this number of sentences will be padded with 0's



# MODEL BUILDING

NET_TYPE = 'dense'  # type of neural net to build
		    # valid options are: 'dense' for fully-connected neural net
		    #                    'conv' for convolutional neural net
		   
OUTER_FOLDS = 3  # number of folds in outer layer of nested cv
INNER_FOLDS = 3  # number of folds in inner layer of nested cv

BATCH_SIZE = 5  

# Describes the types of features that will be created.
# Ex. Suppose FEATURES = ['mrc', 'pronouns', 'all']
# First, a model just using mrc features will be examined in inner cv.
# Then a model just using pronoun features will be examined. 
# Finally, a model with both mrc and pronoun features will be examined.
FEATURES = ['mrc', 'pronouns', 'sentiment', 'filled_pauses', 'turn taking', 'all']

# Decide on the parameters to test the model with
if NET_TYPE == 'dense':

        PARAM_GRID = {
                'num_layer': [2, 3, 4, 5],
                'neurons_per_layer': [50, 500, 1500],
                'perc_dropout': [0, 0.2],
                'optimizer': ['rmsprop', 'Adam'],
                'learning_rate': [1e-4, 1e-2],
                'act_func': ['relu'],
                'epochs': [25, 100, 200]
        }

elif NET_TYPE == 'conv':

        PARAM_GRID = {
                'num_layer': [2, 3],
                'filters': [50, 250],
                'kernel_size': [3, 7],
                'strides': [1],
                'padding': ['valid'],  # ['same', 'valid'],
                'act_func': ['relu'],
                'pool_type': ['MaxPooling1D', 'GlobalMaxPooling1D'], 
			      #'AveragePooling1D', 'GlobalAveragePooling1D'],

                'pool_size': [2],
                'pool_padding': ['valid'],  # ['valid', 'same'],
                'neurons_per_layer': [10, 50],

                'optimizer': ['rmsprop'],  # ['rmsprop', 'Adam'],
                'epochs': [15, 50]
        }
