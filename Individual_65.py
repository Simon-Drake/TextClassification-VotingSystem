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

cur_dir = "/home/simon/Documents/Software/COM3610/"


# Global function to write to output files
def writeToResults(name, s, c):

	results = list()
	with open("Individual_65_{}_{}.txt".format(name, c), "r") as f:
		results = f.readlines()
	
	results.append(s)
	
	with open("Individual_65_{}_{}.txt".format(name, c), "w") as f:
		for i in results:
			f.write(i)

#Data class, handles preprocessing data and distributing it
class Data:
	def __init__(self):
		self.test_classes_topics = dict()

	# Callable tokeniser for the vectorizer
	# Uses Lancaster stemmer and PubMed stop list
	# Returns a list of tokens
	def tokeniser(self, text):

		# removes punctuation
		table = str.maketrans({key: None for key in string.punctuation})
		text = text.translate(table).lower()

		#removes a word if it is shorter than 2 characters, contains a digit or is a stopword. 
		ssw = self.getStopWords()
		stemmer = nltk.stem.lancaster.LancasterStemmer()
		text = text.split(" ")
		filtered = [i for i in text if not(len(i) < 2 or any(j.isdigit() for j in i) or i in ssw)]
		return [stemmer.stem(i) for i in filtered]


	# Return the set of PubMed stop words
	def getStopWords(self):

		sw = open(cur_dir + "Pubmed_stopwords.txt", "r").readlines()
		for i, item in enumerate(sw):
			sw[i] = sw[i].strip()
		ssw = set(sw)
		return ssw

	# Vectorises the data, stores the training data and test data in object
	def vectorise(self, train_corpus, test_corpus):

		# vectorizer = CountVectorizer(tokenizer = tokeniser)
		vectorizer = TfidfVectorizer(tokenizer = self.tokeniser)
		X = vectorizer.fit_transform(train_corpus)
		self.training_data = X.toarray()

		# Create vector of features for unseen instances
		Xt = vectorizer.transform(test_corpus)
		self.test_data = Xt.toarray()

	# This function prepares the test set and train set
	def prepare(self, topic, c):

		# read in relevances for all topics
		with open(cur_dir + "TrainingData/qrels/qrel_abs", "r") as qrel:
			rel = qrel.readlines()

		rtrvdStds = os.listdir(cur_dir  + "TrainingData/{}/".format(topic))

		# get relevance scores for all studies in a topic
		relevance_scores = dict()
		for i in rel:
			if topic in i:
				s = i.strip()
				relevance_scores[s[16:-1].replace(" ", "")] = s[29:]

		# take out unlabelled studies from the data
		rtrvdStds = [k[:-4] for k in rtrvdStds if  k[:-4] in relevance_scores]

		# taking 65% of the data at random
		ntrns = math.floor(len(rtrvdStds) * 0.65)
		indexes = random.sample(range(0, len(rtrvdStds)-1), ntrns)


		train_corpus = list()
		test_corpus = list()
		self.train_classes = list()
		self.test_classes = list()

		for i, item in enumerate(rtrvdStds):

			# read in abstract, title and MeSH terms
			at = open(cur_dir + "TrainingData/{}/".format(topic) + item + ".txt", "r").readlines()[:-1]
			mt = open(cur_dir + "TrainingData/encoded/{}/".format(topic) + item + ".txt", "r").readlines()

			# concatenates title, abstract and encoded MeSH terms to make one string (all tokens have same weights)
			tam = at[0].strip() + " " + at[1].strip() + " " + mt[0]

			# Adding unprocessed text to training or testing corpus and corresponding label
			if i in indexes:
				train_corpus.append(tam)
				self.train_classes.append(int(relevance_scores[item]))
			else:
				test_corpus.append(tam)
				self.test_classes.append(int(relevance_scores[item]))

		self.test_classes_topics[topic] = self.test_classes
		self.vectorise(train_corpus, test_corpus)
		

		# Only preprocesses data once, data is dumped into a pickle file for subsequent classifiers
		f = open(cur_dir + "PickledData" + "/65_{}_{}.pkl".format(topic, c), 'wb')
		pickle.dump(self, f)
		f.close()


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

		s = str(round((start_time_topic - time.time())/-60, 4))

		self.test_classes = dt.test_classes
		f = open(cur_dir + "PickledData" + "/65_{}_{}_{}.pkl".format(self.name, topic, c), 'wb')
		pickle.dump(self, f)
		f.close()

		ev.evaluate(self.predict, topic, self.name, dt.test_classes, c, s)

		
# Evaluater deals with using predicitons and labels to get all relevant metrics
class Evaluater:
	def __init__(self):
		self.total_recall = 0

	# Writes the results of each classification to the classifier's txt file
	# Metrics : Recall, Precision, f5, accuracy, and SO
	def evaluate(self, predict, topic, name, test_css, c, s):

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

		writeToResults(name, "{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | {5:10} |\n".format(topic, str(recall), str(precision), str(f5), str(so), str(s)), c)


if __name__ == "__main__":

	# test ts
	topics = ["CD008760", "CD010705", "CD010542"]
	# real ts
	# topics = ["CD009519", "CD011984", "CD011975"]

	# Declare classifiers and their names
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


	dt = Data()
	for c in range (0, 5):
		for n, j in enumerate(classifiers):
			#time each classifier
			start_time = time.time()

			ev = Evaluater()
			cf = Classifier(j, names[n])
			print("doing classifier...{}".format(names[n]))

			# Prepare the results file
			with open("Individual_65_{}_{}.txt".format(names[n], c), "w") as f:
				f.write("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | {5:10} | \n".format("Topic", "Recall", "Precision", "F5", "SO", "Time"))

			for i in topics:

				print("processing topic... ", i)

				# If the topic has already been preprocessed, load it
				if "65_{}_{}.pkl".format(i, c) in os.listdir(cur_dir + "PickledData"):
					f = open(cur_dir + "PickledData" + "/65_{}_{}.pkl".format(i, c), 'rb')
					dt = pickle.load(f)
					f.close()
				else:
					dt.prepare(i, c)

				cf.predict = list()		
				# Predictions handled by the classifier class	
				cf.classify(dt, i, ev, [], c)

			s = "Time Taken :" + str(round((start_time - time.time())/-60, 4)) + "\n"
			writeToResults(names[n], s, c)

	# The following loops use what was written in the individual files (for each validation run)
	# and averaged them to make cross validated stats
	for i in names:
		with open("cross_validated_65_{}".format(i), "w") as f:
			f.write("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | {5:10} | \n".format("Topic", "Recall", "Precision", "F5", "SO", "Time"))

	for i in names:

		# Stats is a dict {topic : {metric: value}}
		stats = dict()
		for t in topics:
			stats[t] = {"recall": 0, "precision": 0, "F5": 0, "SO":0, "time":0}
		average = {"recall": 0, "precision": 0, "F5": 0, "SO":0, "time":0}

		# Read each file for the validation run and classifier
		for j in range(0,5):
			with open("Individual_65_{}_{}.txt".format(i, j), "r") as f:
				r = f.readlines()

			# Retrieve data
			for n in r[1:-1]:
				x = n.split(":")
				topic = x[0].strip()
				x = x[1].split("|")[:-1]
				stats[topic]["recall"] += float(x[0].strip())
				stats[topic]["precision"] += float(x[1].strip())
				stats[topic]["F5"] += float(x[2].strip())
				stats[topic]["SO"] += float(x[3].strip())
				stats[topic]["time"] += float(x[4].strip())

		# Average data
		for t in topics:
			stats[t]["recall"] = round(stats[t]["recall"]/5, 3)
			average["recall"] += stats[t]["recall"]
			stats[t]["precision"] = round(stats[t]["precision"]/5, 3)
			average["precision"] += stats[t]["precision"]
			stats[t]["F5"] = round(stats[t]["F5"]/5, 3)
			average["F5"] += stats[t]["F5"]
			stats[t]["SO"] = round(stats[t]["SO"]/5, 3)
			average["SO"] += stats[t]["SO"]
			stats[t]["time"] = round(stats[t]["time"]/5, 3)
			average["time"] += stats[t]["time"]

			# Write cross validated results to result file
			results = list()

			with open("cross_validated_65_{}".format(i), "r") as f:
				results = f.readlines()
		
			results.append("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | {5:10} |\n".format(t, str(stats[t]["recall"]), str(stats[t]["precision"]), str(stats[t]["F5"]), str(stats[t]["SO"]), str(stats[t]["time"])))
			with open("cross_validated_65_{}".format(i), "w") as f:
				for r in results:
					f.write(r)

		# Write the average from all topics and cross validations 
		for x in average:
			average[x] = round(average[x]/len(topics), 3)
		with open("cross_validated_65_{}".format(i), "r") as f:
			results = f.readlines()
		results.append("{0:10} : {1:10} | {2:10} | {3:10} | {4:10} | {5:10} |\n".format("average :", str(average["recall"]), str(average["precision"]), str(average["F5"]), str(average["SO"]), str(average["time"])))
		with open("cross_validated_65_{}".format(i), "w") as f:
			for r in results:
				f.write(r)

		