PURPOSE

The goal of this program was two-fold: 

	1) Create a general program that can train and compare various neural networks and baseline models on the user's dataset.

	2) Use nested cross-validation to find which combination of architectures, hyperparameters, and features give the best predictive performance.



STRUCTURE 

create_dataset.py - The first program the user should run. Generates a text file
		    containing all the features that can be used to train the model.

main.py - The only other program the user needs to run directly. Uses nested cv to
	  train and compare various neural networks and baseline models. Generates a 
	  text file containing brief description of results.

const.py - Allows the user to modify the behaviour of the program. Very important!
	   Often you will see variables with UPPERCASE NAMES. These are variables that have 
	   been defined in this file.

data/ - Contains various data. The exact contents of this directory are determined by
	how the user modifies the const.py file.

header_files/ - Contains various function definitions 

results/ - Holds the result files created by main.py



GENERATING FEATURES

There are many ways to generate features from a transcript (which is essentially a list of words).
I now give an overview and examples of how I created word features.

The create_dataset.py program creates sentence-level features from a directory of text files.
This means that each sentence in a transcript gets its own feature.

For each text file in this directory, and for each type of score column in the text file, 
and for each sentence in a transcript, the scores of the words that are in both the text file 
and the sentence are summed together. This sum is one feature for that transcript.

See FEATS_DIR/mrc/ for an example of such a directory of text files.


                                EXAMPLE 

In a certain directory, suppose we have a single text file that looks like this:

	WORD            SCORE1          SCORE2
	
	FIRST           10              -2
	MEETING         -               2
	OPERATE         5               -
	PROJECT         -3              5

And suppose we have two transcripts that look like this:

	ES2002a.B.dialog-act.dharshi.1  Okay
	ES2002a.B.dialog-act.dharshi.3  This is the first meeting for our first project.
	
	ES2002b.B.dialog-act.dharshi.1  First!
	ES2002b.B.dialog-act.dharshi.2  Okay .
	ES2002b.B.dialog-act.dharshi.3  Sorry ?
	ES2002b.B.dialog-act.dharshi.4  Okay , everybody all set to start the meeting ?

We will first look at the first transcript

	In the first sentence, none of FIRST, MEETING, OPERATE, or PROJECT appear.
	Thus, this sentence has a SCORE1 of 0.
	
	In the second sentence, MEETING and PROJECT appear once, and FIRST appears twice
	Thus, this sentence has a SCORE1 of 17
	(Captilization does not matter, and words without a score effectively have a score of 0).
	
	The first sentence will have a SCORE2 of 0, while the second has a SCORE2 of 3.

Look now at the second transcript

	The first sentence has a SCORE1 of 10 and a SCORE2 of -2.
	The second and third sentences both have a SCORE1 and SCORE2 of 0.
	The last sentence has a SCORE1 of 0 and a SCORE2 of 2.  


                        POSSIBLE REPRESENTATION

We will have a dataframe that contains one row for each transcript:

	          |---- SCORE1 ----|  |---- SCORE2 ----|
	MEETINGS  FEATURE1  FEATURE2  FEATURE3  FEATURE4
	
	ES2002a   0         17        0         3       
	
	
	          |------------- SCORE1 ---------------|  |------------- SCORE2 ---------------|  
	MEETINGS  FEATURE1  FEATURE2  FEATURE3  FEATURE4  FEATURE5  FEATURE6  FEATURE7  FEATURE8
	
	ES2002b   10        0         0         0         -2        0         0         2              


                                PROBLEMS

This is no good though, since each row of our dataframe must have the same number of columns
This is occuring because our transcripts have a different number of sentences. 

Thus, if there are fewer than NUM_FEAT sentences in the transcript, the remaining sentences
are padded with 0's 

We don't want our dataframe to have too many columns though.
Thus, if there are more than NUM_FEAT sentences in the transcript, only the first NUM_FEAT 
sentences are used to create the features.


                        BETTER REPRESENTATION

Our dataframe will now look like this:

	          |-------- SCORE1 ----------|  |-------- SCORE2 ----------|
	MEETINGS  FEATURE1  FEATURE2  FEATURE3  FEATURE4  FEATURE5  FEATURE6  Response
	
	ES2002a   0         17        0         0         3         0         (Not relevant)
	ES2002b   10        0         0         -2        0         0         (Not relevant)



                            FORMAT OF FILES

The directory of data files must contain one or more text files.

The first column of each file must contain the words that each sentence will be searched for.
All of these words must be only one word (ex. 'excuse me' is not a valid word).

All subsequent columns (if any) must be scores for those words.

In each file, columns must already have column names. 
If the directory has multiple files (ex. FEATS_DIR/sentiment), all score columns should have UNIQUE NAMES
  
This ends the discussion on feature derivation.



KNOWN ISSUES

This program will not run on Windows, because files are referenced using '/'.
This is probably an easy fix, but I did not have time to look into it.

If you are running out of memory when running the convolutional net, try making the network smaller.
In particular, it may be helpful to reduce the filters and neurons_per_layer. 
Reducing BATCH_SIZE can also help.
If none of these work, you can remove 'all' from FEATURES. This will prevent all the features from 
being used in the same model (they will still be used individually).



SUGGESTED IMPROVEMENTS

Currently, the results text file generated by main.py is not very detailed. It would be easy 
to add more info (score info other than MSE, standard deviation, etc).



VERSION OF SOFTWARE USED

Linux Ubuntu 18.10
Python 3.7.1

I also used the package manager Anaconda, which I highly recommend.


Any questions regarding the functionality of this program can be directed to 
jonathan.fitz@student.ufv.ca

