*****************************************************WARNING**********************************************************
Individual.py and individual_65.py can take up A LOT of memory depending on the topics chosen so make sure you have at
least 25GB (with a margin to be safe) free for either of them if using topics that amount to more than 20 000 PMIDS
**********************************************************************************************************************


These programs read and write to many files. Most of the results are printed in the current directory. 
Pickled data is in a subdirectory named "PickledData"
Topics are in a subdirectory named "TrainingData"
Relevance files and encoded MeSH terms are in a subdirectory of TrainingData. 

----------------------------------------------------------------------------------------------------------------------

Optimisation stage is done using "Grid_optimisation.py"
You need to define the classifiers and their parameters which don't change in the "classifiers" array.
Specify which topic you use at the beginning of the main method. 
In the "parameter_grid" array, you specify the parameters to test as documented by sklearn's GridSearchCV library. 
The program writes the mean test results for all the parameters outlining the best combinations in a file named :
"cvresults_CLASSIFIERNAME.txt"

----------------------------------------------------------------------------------------------------------------------

The Individual runs of the classifiers are done in either Individual.py, Individual_65.py or Undersample.py depending 
on the sampling method. These files are largely the same except from the "prepare" method which samples the data. 
All you need to do is define the classifiers in the classifiers array and the topics in the topics array at the
start of the main method. 
The program then runs the cross validations and writes the results to files in the following manner: 

For each validation: 

FILENAME_CLASSIFIERNAME_VALIDATIONRUN.txt

cross validated files are written in either: 
cross_validated_CLASSIFIERNAME.txt
or
cross_validated_65_CLASSIFIERNAME.txt
or
cross_validated_undersample_CLASSIFIERNAME.txt

----------------------------------------------------------------------------------------------------------------------
For each file used for the voting system you need to specify THE SAME "topics" array, "classifiers" aray and "names" 
array as used in the individual runs. This way the program can unpack the right pickle files. The voting scripts
use the predictions for certain topics and validation runs for different algorithms to form combined predictions and
evaluates them. 

The results are written to: 

Voterz_VALIDATIONRUN.txt
Voterz_65_VALIDATIONRUN.txt
Voterz_undersample_VALIDATIONRUN.txt

Cross validated to:

** Important ** : The code does not yet print the names of the classifiers on the cross
validated files, to see which algorithms produced the results you have to compare with a 
one of the file types above. 

cross_validated_voterz.txt
cross_validated_65_voterz.txt
cross_validated_voterz_undersample.txt

----------------------------------------------------------------------------------------------------------------------

The results from the experiments presented in my dissertation are saved in folders with manifested names. 