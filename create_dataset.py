# This program creates sentence-level features from a directory of text files.
# This means that each sentence in a transcript gets its own feature.

# See the README file for an in-depth explanation of how these features are derived.

import numpy as np
import pandas as pd
import os

from header_files.rm_missing_scores import rm_missing_scores
from header_files.turn_taking import turn_taking
import const


pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)

print('Calculating...\n')

# Return dictionary of meetings where at least one participant has answered question
# Value of key is sum of scores for that meeting
meet_scores = rm_missing_scores()
meets = list(meet_scores.keys())

#print(meets)
#print(len(meets))

# Record all the words in all the transcripts, including the transcript and sentence where it was found
word_sent_list = []  

for trans in os.listdir(const.TRANS_DIR):

	if trans.split('.')[0] in meets:

		f = open(const.TRANS_DIR+'/'+trans, 'r')
		scan = f.readlines()
		f.close()

		sent_num = 1

		for sent in scan[2:]:    
			for word in sent.split()[1:]:
				row = {'Word': word.lower(), 'Meeting': trans.split('.')[0], 
				       'Sentence': sent_num}			
				word_sent_list.append(row)

				sent_num = sent_num + 1

# Convert to dataframe. This is faster than if we had continuously appended to df
word_sent_df = pd.DataFrame(word_sent_list, columns=['Word', 'Meeting', 'Sentence'])

# Keep only words that correspond to a sentence number less than or equal to NUM_FEAT
word_sent_df = word_sent_df.loc[word_sent_df["Sentence"] <= const.NUM_FEAT, :]

# For each word in any transcript, find the score for every word feature
unique_words = pd.unique(word_sent_df['Word'])

word_scores_list = []

# Record combinations of Feature Types and Score Types that appear
valid_feat_scores = []

for feat_dir in os.listdir(const.FEATS_DIR):

	for feat_file in os.listdir(const.FEATS_DIR+'/'+feat_dir):

		feat = pd.read_table(const.FEATS_DIR+'/'+feat_dir+'/'+feat_file, delim_whitespace=True, 
				     comment='#', encoding = "ISO-8859-1")

		# Clean up dataset
		feat = feat.replace('-', 0)
		feat.rename(columns={feat.columns[0]: 'Word'}, inplace=True)
		feat['Word'] = feat['Word'].str.lower()	
	
		# If feat has only one column (ie. no score column), give it a column of 1's
		# Counting the score of each word is then just counting number of occurences
		if feat.shape[1] == 1:
			feat['Count'] = [1]*feat.shape[0]

		feat = feat.astype({score_col: 'float16' for score_col in feat.columns[1:]})

		for col in feat.columns[1:]:
			valid_feat_scores.append([feat_dir, col])

		# Retrieve the scores corresponding to the transcript word, if they exist
		for word in unique_words:
			scores = feat.loc[feat['Word'] == word, feat.columns[1]:]

			if scores.empty: continue

			for i in list(range(scores.shape[1])):
				row = {'Word': word, 'Feature Type': feat_dir,
				       'Score Type': scores.columns[i], 'Score': scores.iloc[0, i]}
				word_scores_list.append(row)

# As before, convert to dataframe
word_scores_df = pd.DataFrame(word_scores_list, columns=['Word', 'Feature Type', 'Score Type', 'Score'])

# Remove words that have a score of 0
# This saves memory, but causes additional work below
word_scores_df = word_scores_df.loc[word_scores_df['Score'] != 0, :]

# Combine these dataframes
combined = word_scores_df.merge(word_sent_df, copy=False)

# For each meeting, feature type, score type, and sentence, find the sum of the word scores
grouped = combined.groupby(['Meeting', 'Feature Type', 'Score Type', 'Sentence']).aggregate(np.sum)

# Rearrange to create the final dataset
dataset = grouped.unstack(fill_value=0)

# Touch it up a bit
dataset.columns = dataset.columns.droplevel()
dataset.rename_axis(None, axis=1, inplace=True)
dataset = dataset.astype(dtype='int16')
#dataset = dataset.round(1)
dataset = dataset.reset_index()

# Notice that dataset only shows transcripts/feature types/score types/sentences that have at least 
# one non-zero word score. Thus, if all the words in the transcript ES2002a score a zero for all 
# feature types, ES2002a will not appear as a row in our dataframe.
# If ES2002a is a meeting with at least one questionnaire response though, then this is bad since 
# we want it in our dataset. So we now have to do some additional work

# If the columns 1, 2, ... , NUM_FEAT don't all exist, add missing column and fill with 0's
for x in list(range(1, const.NUM_FEAT+1)):
	if x not in dataset.columns:
		dataset[x] = [0]*dataset.shape[0]

# If a row does not exist for some combination of valid meetings and word scores, add it to dataset
missing = []
for meet in meets:
	for e in valid_feat_scores:
			scores = dataset.loc[(dataset['Meeting'] == meet) &
					     (dataset['Feature Type'] == e[0]) &
					     (dataset['Score Type'] == e[1])]

			if scores.empty: 
				row = {'Meeting': meet, 'Feature Type': e[0], 'Score Type': e[1]}
				row.update({num: 0 for num in list(range(1, const.NUM_FEAT+1))})

				missing.append(row)

dataset = dataset.append(missing, ignore_index=True)

# Add turn-taking features to model
dataset = dataset.append(turn_taking(meets), ignore_index=True)

#print(list(dataset.columns))
#print(dataset.iloc[:6, :6])
#print(dataset.shape)

# Write this dataset to file
dataset.to_csv(const.DATASET)
