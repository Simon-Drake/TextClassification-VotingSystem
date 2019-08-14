import string, os, nltk.stem, math, pickle, time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt 
import random
import itertools

cur_dir = "/home/simon/Documents/Software/COM3610/"


# Global function to write to output files
def writeToResults(s, c):
	results = list()
	with open("Voterz_65_{}.txt".format(c), "r") as f:
		results = f.readlines()
	
	results.append(s)
	
	with open("Voterz_65_{}.txt".format(c), "w") as f:
		for i in results:
			f.write(i)


# Classifier class deals with training classifiers and making predictions
class Classifier:
	def __init__(self, classifier, name):
		self.classifier = classifier
		self.name = name

	# Train classifier
	def classify(self, dt, topic, ev, sw, c):

		start_time_topic = time.time()

		# Fit data, if sample weight given use it
		if sw == []:
			self.classifier.fit(dt.training_data, dt.train_classes)
		else:
			self.classifier.fit(dt.training_data, dt.train_classes, sample_weight=sw)

		# Make predictions
		for i in range(len(dt.test_classes)):
			self.predict.append(self.classifier.predict(dt.test_data[i:i+1])[0])

		s = str(round((start_time_topic - time.time())/-60, 3))

		ev.evaluate(self.predict, topic, self.name, dt.test_classes, c, s)

		
# Evaluater deals with using predicitons and labels to get all relevant metrics
class Evaluater:

	# Writes the results of each classification to the classifier's txt file
	# Metrics : Recall, Precision, f5, accuracy, and SO
	def evaluate(self, predict, topic, test_css, c):

		tp = 0
		tn = 0
		fp = 0 
		fn = 0 

		# Uses if bigger than or equal to because it will process a list of votes
		for i, item in enumerate(predict):
			if item >= 1 and test_css[i] == 1:
				tp += 1
			elif item >= 1 and test_css[i] == 0:
				fp += 1
			elif item == 0 and test_css[i] == 0:
				tn += 1
			else:
				fn += 1

		# Uses TP, FP, TN and FN to calculate metrics and write to result file. 
		relevant = tp + fn
		non_relevant = fp + tn
		positives = tp + fp
		negatives = tn + fn
		correct = tn + tp

		if relevant == 0 or tp==0:
			recall = 0
			precision = 0
			accuracy = round(correct/(relevant+non_relevant), 6)
			f5 = 0.0
		else:
			recall = round(tp/relevant, 6)
			precision = round(tp/positives, 6)
			accuracy = round(correct/(relevant+non_relevant), 6)
			f5 = round( ((5**2 + 1)*tp) / ( ((5**2 + 1)*tp) + fp + (5**2)*fn ), 6)


		so = round(negatives/(non_relevant+relevant), 6)

		writeToResults("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} |\n".format(topic, str(recall), str(precision), str(f5), str(so)), c)

# This function combines the predictions into one and evaluates it. 
def write_voters(x, dict_c, voters, ev, test_classes, i):

	votes = dict()
	for t in topics:
		votes[t] = list()
		
	# Uses itertools to get unique combinations of voters
	for comb in x:
		s = ""
		for j in comb:
			s+= dict_c[j] + " "
		writeToResults(s+"\n",i)

		# For each classifer in the combination add the votes
		for j in comb:
			for t in topics:
				if len(votes[t])==0:
					votes[t] = voters[i][dict_c[j]][t]
				else:
					votes[t] = [k+s for k,s in zip(votes[t], voters[i][dict_c[j]][t])]

		for t in topics:
			ev.evaluate(votes[t], t, test_classes[i][t], i)

		votes = dict()
		for t in topics:
			votes[t] = list()

# Function to write stats to cross validated results file
def write_stats_cv(t, stats):
	results = list()

	with open("cross_validated_65_voterz", "r") as f:
		results = f.readlines()

	results.append("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} |\n".format(t, str(stats[t]["recall"]), str(stats[t]["precision"]), str(stats[t]["F5"]), str(stats[t]["SO"])))
	with open("cross_validated_65_voterz", "w") as f:
		for r in results:
			f.write(r)

# Function to write to cross validated results file
def write_cv(s):
	results = list()

	with open("cross_validated_65_voterz", "r") as f:
		results = f.readlines()

	results.append(s)
	with open("cross_validated_65_voterz", "w") as f:
		for r in results:
			f.write(r)


if __name__ == "__main__":

	# test ts
	topics = ["CD008760", "CD010705", "CD010542"]
	# real ts
	# topics = ["CD009519", "CD011984", "CD011975"]



	# Define the classifiers used and their names 
	classifiers = [
		ComplementNB(alpha=0.0075),

		AdaBoostClassifier(n_estimators=500, algorithm='SAMME.R'),

		MLPClassifier(hidden_layer_sizes=(250,), max_iter=600, solver='lbfgs'),

		KNeighborsClassifier(n_neighbors=3, weights='distance'),

		SVC(C=0.05, class_weight='balanced', tol=0.00005, kernel='linear')
	]


	names = [
		"ComplementNB",

		"AdaBoostClassifier",

		"MLPClassifier",

		"KNeighborsClassifier",

		"SVC"
	]

	# Dict used to find combinations of classifiers 
	dict_c = {'A':"ComplementNB",

		'B':"AdaBoostClassifier",

		'C':"MLPClassifier",

		'D':"KNeighborsClassifier",

		'E':"SVC"}


	# Voters is a nested dictionary {run :{classifier : {topic : predictions}}}
	voters = dict()
	test_classes = dict()

	for c in range (0, 5):
		voters[c] = dict()
		test_classes[c] = dict()
		for n, j in enumerate(classifiers):
			ev = Evaluater()
			voters[c][names[n]] = dict()
			print("doing classifier...{}".format(names[n]))


			for i in topics:

				print("processing topic... ", i)

				# Load the predictions of the classifier at a certain validation run for a topic
				f = open(cur_dir + "PickledData" + "/65_{}_{}_{}.pkl".format(names[n], i, c), 'rb')
				cl = pickle.load(f)
				f.close()
				
				voters[c][names[n]][i] = cl.predict
				test_classes[c][i] = cl.test_classes



	# Voters is a nested dictionary {run :{classifier : {topic : predictions}}}
	ev = Evaluater()
	for i in range(0, 5):

		with open("Voterz_65_{}.txt".format(i), "w") as f:
			f.write("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | \n".format("Topic", "Recall", "Precision", "F5", "SO"))

		votes = dict()
		for t in topics:
			votes[t] = list()

		# Add the votes of all the classifiers (this secion only deals with 5 voters)
		for j in voters[i]:
			for t in topics:
				if len(votes[t])==0:
					votes[t] = voters[i][j][t]
				else:
					votes[t] = [k+s for k,s in zip(votes[t], voters[i][j][t])]

		for t in topics:
			ev.evaluate(votes[t], t, test_classes[i][t], i)

		# Get the votes for each combination of classifier
		writeToResults("\n----------4 voters----------\n", i)
		x = itertools.combinations('ABCDE', 4)
		write_voters(x, dict_c, voters, ev, test_classes, i)
		writeToResults("\n----------3 voters----------\n", i)
		x = itertools.combinations('ABCDE', 3)
		write_voters(x, dict_c, voters, ev, test_classes, i)
		writeToResults("\n----------2 voters----------\n", i)
		x = itertools.combinations('ABCDE', 2)
		write_voters(x, dict_c, voters, ev, test_classes, i)

			


	
	with open("cross_validated_65_voterz", "w") as f:
		f.write("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | \n".format("Topic", "Recall", "Precision", "F5", "SO"))

	# This function uses the votes from each validation run to find cross validated stats for the voting system 
	def compute_stats(y,z):

		# Stats is a nested dictionary {topic {metric: value}}
		stats = dict()
		for t in topics:
			stats[t] = {"recall": 0, "precision": 0, "F5": 0, "SO":0}
		average = {"recall": 0, "precision": 0, "F5": 0, "SO":0}
		for j in range(0,5):
			with open("Voterz_65_{}.txt".format(j), "r") as f:
				r = f.readlines()
			
			# Retrieve data
			for n in r[y:z]:
				x = n.split(":")
				topic = x[0].strip()
				x = x[1].split("|")[:-1]
				stats[topic]["recall"] += float(x[0].strip())
				stats[topic]["precision"] += float(x[1].strip())
				stats[topic]["F5"] += float(x[2].strip())
				stats[topic]["SO"] += float(x[3].strip())

		# Get average
		for t in topics:
			stats[t]["recall"] = round(stats[t]["recall"]/5, 3)
			average["recall"] += stats[t]["recall"]
			stats[t]["precision"] = round(stats[t]["precision"]/5, 3)
			average["precision"] += stats[t]["precision"]
			stats[t]["F5"] = round(stats[t]["F5"]/5, 3)
			average["F5"] += stats[t]["F5"]
			stats[t]["SO"] = round(stats[t]["SO"]/5, 3)
			average["SO"] += stats[t]["SO"]
			write_stats_cv(t, stats)
		for x in average:
			average[x] = round(average[x]/len(topics), 3)
		write_cv("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} |\n".format("average :", str(average["recall"]), str(average["precision"]), str(average["F5"]), str(average["SO"])))
		
	# This section write to the cross validation file for all voting runs
	compute_stats(1,4)
	startindex = 7
	write_cv("\n----------4 voters----------\n\n")
	for i in range(5):
		compute_stats(startindex, startindex+3)
		startindex += 4
		write_cv("\n")
	startindex = 29
	write_cv("\n----------3 voters----------\n\n")
	for i in range(10):
		compute_stats(startindex, startindex+3)
		startindex += 4
		write_cv("\n")
	startindex = 71
	write_cv("\n----------2 voters----------\n\n")
	for i in range(10):
		compute_stats(startindex, startindex+3)
		startindex += 4
		write_cv("\n")


	

