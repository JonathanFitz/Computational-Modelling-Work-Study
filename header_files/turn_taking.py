# Creates turn-taking features. This is mostly Dr. Murray's code.

import re, os
import numpy as np
import pandas as pd

import const

# Take as input list of transcripts that have a corresponding response value
def turn_taking(meets):

	turn_list = []

	for filename in os.listdir(const.TRANS_DIR):
		if filename.split('.')[0] in meets:

			f = open('{}/{}'.format(const.TRANS_DIR, filename), 'r')
			scan = f.readlines()
			f.close()
			#print(filename)

			#meetid = re.findall('[a-z]+[0-9]*_[A-Z][A-Z]\d\d\d\d[a-z]', 
			#	             '{}/{}'.format(const.TRANS_DIR, filename))[0]
			#print(meetid)
			
			part_dict = {}
			
			feas = []
			
			counter = 0
			for eachline in scan:
				if 'dialog-act' in eachline:
					counter += 1
					part_id = re.findall('\.([A-Z])\.', eachline)[0]
					if part_id not in part_dict:
						feas.append(str(counter))
						part_dict[part_id] = counter
					else:
						feas.append(str(counter - part_dict[part_id]))
						part_dict[part_id] = counter
			            
			#print(len(feas))
			if len(feas) > const.NUM_FEAT:
				feas = feas[0:const.NUM_FEAT]
			    
			if len(feas) < const.NUM_FEAT:
				zeds = ['0'] * (const.NUM_FEAT - len(feas))
				feas = feas + zeds
			
			row = {'Meeting': filename.split('.')[0], 'Feature Type': 'turn taking', 
			       'Score Type': 'turns'}
			row.update({n+1: feas[n] for n in list(range(const.NUM_FEAT))})

			turn_list.append(row)

	#print(turn_list)
			
	turn_df = pd.DataFrame(turn_list, columns=['Meeting', 'Feature Type', 'Score Type']+
						  [n+1 for n in list(range(const.NUM_FEAT))])

	#print(turn_df.head())
	#print(turn_df.shape)

	return turn_df	
