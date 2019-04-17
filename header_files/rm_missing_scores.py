# Return only the transcript files for which at least one participant answered 
# the questionnaire question specified in const.py. 
# For each of these selected files, also return the sum of the answers (scores).

import numpy as np
import pandas as pd
import os, re

import const

def rm_missing_scores():

        # Create list of files in transcript directory 
        trans = sorted(os.listdir(const.TRANS_DIR))

        meet_scores = {}  # Map transcripts that have a score to their total score
        df_q = pd.read_csv(const.MEET_SCORES)

        # Go through each transcript name in turn
        for meet in trans:

                # Only keep the first part of the filename
                meet_name = re.match('[A-Z][A-Z]\d\d\d\d[a-z]', meet)[0]
                #print(meet_name)

                meet_pre = re.sub('[a-z]', '', meet_name)
                #print('\nmeet_pre is: ')
                #print(meet_pre)

                meet_suff = re.sub('[A-Z][A-Z]\d\d\d\d', '', meet_name)
                #print('\nmeet_suff is: ')
                #print(meet_suff)

                meet_map = {'a': '2', 'b': '4', 'c': '6', 'd': '8'}
                #print('\nmeet_map is: ')
                #print(meet_map)

                sub_sat = df_q[df_q['HTMLID'] == const.RESPONSE]
                #print('\nsub_sat is: ')
                #print(sub_sat)

                meet_sat = sub_sat[(sub_sat['conv'] == meet_pre) &
                                   (sub_sat['QuestionnaireID'] == meet_map[meet_suff])]
                #print('\nmeet_sat is: ')
                #print(meet_sat)

                scores = meet_sat['Answer']
                scores = [float(s) for s in scores]

                # Check if at least one participant answered this question
                if len(scores) > 0:
                        #print('\nscores is: ')
                        #print(scores)

                        meet_scores[meet_name] = np.sum(scores)

        return meet_scores
