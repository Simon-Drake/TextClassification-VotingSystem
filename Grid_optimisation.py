import string, os, nltk.stem, math, pickle, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

cur_dir = "/home/simon/Documents/Software/COM3610/"

#Data class, handles preprocessing data and distributing it
class Data:

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
	def vectorise(self, train_corpus):

		vectorizer = TfidfVectorizer(tokenizer = self.tokeniser)
		X = vectorizer.fit_transform(train_corpus)
		self.training_data = X.toarray()

	# Function to read in the relevant text information
	def getText(self, topic, item, relevance_scores):
		# read in abstract, title and MeSH terms
		at = open(cur_dir + "TrainingData/{}/".format(topic) + item + ".txt", "r").readlines()[:-1]
		mt = open(cur_dir + "TrainingData/encoded/{}/".format(topic) + item + ".txt", "r").readlines()

		# concatenates title, abstract and encoded MeSH terms to make one string (all tokens have same weights)
		tam = at[0].strip() + " " + at[1].strip() + " " + mt[0]


		self.train_corpus.append(tam)
		self.train_classes.append(int(relevance_scores[item]))

	
	def prepare(self, topic):

		# read in relevances for all topics
		with open(cur_dir + "TrainingData/qrels/qrel_abs", "r") as qrel:
			rel = qrel.readlines()

		# read in relevant studies
		rtrvdStds1 = os.listdir(cur_dir + "TrainingData/{}/".format(topic))

		# get relevance scores for all studies in a topic
		relevance_scores = dict()
		for i in rel:
			if topic in i:
				s = i.strip()
				relevance_scores[s[16:-1].replace(" ", "")] = s[29:]

		# take out unlabelled studies from the data
		rtrvdStds1 = [k[:-4] for k in rtrvdStds1 if  k[:-4] in relevance_scores]


		# Get data
		self.train_corpus = list()
		self.train_classes = list()
		for i, item in enumerate(rtrvdStds1):
			self.getText(topic, item, relevance_scores)

		# Create vectors 
		self.vectorise(self.train_corpus)
		

		# Only preprocesses data once, data is dumped into a pickle file for subsequent classifiers
		f = open(cur_dir + "PickledData" + "/{}.pkl".format(topic), 'wb')
		pickle.dump(self, f)
		f.close()


# Classifier class deals with training classifiers and making predictions
class Classifier:
	def __init__(self, classifier, name):
		self.classifier = classifier
		self.name = name

	# Train classifier
	def classify(self, dt, parameter_grid):
		start_time = time.time()
		
		# Define scorer and cross validation method
		ffive_scorer = make_scorer(fbeta_score, beta=5)
		rs = ShuffleSplit(n_splits=5, test_size=.5)
		searcher = GridSearchCV(self.classifier, parameter_grid, scoring=ffive_scorer, cv=rs)

		# Fit data, if sample weight given use it
		searcher.fit(dt.training_data, dt.train_classes)
		
		# Write mean test score for each parameter and best parameters in current directory
		st =""
		for i in zip(searcher.cv_results_['params'], searcher.cv_results_['mean_test_score']):
			for j, item in enumerate(i[0].keys()):
				if j+1 < len(i[0].keys()):
					st += "{0:10}: {1:10} ".format(item, str(i[0][item]))
				else:
					st += "{0:10}: {1:10} score: {2:10} \n".format(item, str(i[0][item]), str(i[1]))

		st += "\nBest params: "			
		for i in searcher.best_params_.keys():
			st += "{0:10}: {1:10}".format(i, str(searcher.best_params_[i]))
		
		st += "\nTime Taken :" + str(round((start_time - time.time())/-60, 2)) + "\n"

		with open("cvresults_{}".format(self.name), "w") as f:
			f.write(st)
		

if __name__ == "__main__":

	# Define topic
	topic = "CD008760"

	# Define classifiers, parameter grids and names
	classifiers = [
		ComplementNB(),

		KNeighborsClassifier(weights='distance'),

		AdaBoostClassifier(algorithm='SAMME.R'),

		MLPClassifier(max_iter=600, solver='lbfgs'),

		SVC(class_weight='balanced', tol=0.00005, kernel='linear')
	]

	parameter_grids = [
		{'alpha':[0.003, 0.0035, 0.004, 0.0045]},
        {'n_neighbors':[1, 2, 3, 4]},
        {'n_estimators':[200]},
        {'hidden_layer_sizes':[240, 250, 260, 270]},
        {'C':[0.75, 0.85, 0.95]}
	]

	names = [
		"ComplementNB()",
		"KNeighborsClassifier()",
		"AdaBoostClassifier(algorithm='SAMME.R')",
		"MLPClassifier(max_iter=500, solver='lbfgs')",
		"SVC(class_weight='balanced', tol=0.00005, kernel='linear')"
	]

	dt = Data()

	for n, j in enumerate(classifiers):

		cf = Classifier(j, names[n])
		print("doing classifier...{}".format(names[n]))

		# If the topic has already been preprocessed, load it
		if "{}.pkl".format(topic) in os.listdir(cur_dir + "PickledData"):
			f = open(cur_dir + "PickledData" + "/{}.pkl".format(topic), 'rb')
			dt = pickle.load(f)
			f.close()
		else:
			dt.prepare(topic)

		# Predictions handled by Classifier class
		cf.classify(dt, parameter_grids[n])

