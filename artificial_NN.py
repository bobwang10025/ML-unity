import numpy as np

def documents(speechdoc):
	return list(speechdoc.reviews())
def continuous(speechdoc):
	return list(speechdoc.scores())
def make_categorical(speechdoc):
	"""
	terrible:   0.0 <y <= 1.0
	medium ok:  1.0 <y <= 2.0
	ok:         2.0 <y <= 3.0
	great:      3.0 <y <= 4.0
	amazing:    4.0 <y <= 5.0
	"""
	return np.digitize(continuous(speechdoc), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

def train_model(path, model, continuous=True, storage_disk=None, cv=12):
	"""
	train model, construct cross-validation scores, fit model and
	return scores
	"""
	# load data and label it
	speechdoc = PickledReviewsReader(path)
	X = documents(speechdoc)
	if continuous:
		y = continuous(speechdoc)
		scoring = 'r2_score'
	else:
		y = make_categorical(speechdoc)
		scoring = 'f1_score'
	# compute scores
	scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
	# save it
	if storage_disk:
		joblib.dump(model, storage_disk)
	# fit model
	model.fit(X,y)
	# return scores
	return scores

if __name__ == '__main__':
	from transformer import TextNormalizer, identity
	from reader import PickledReviewsReader
	
	from sklearn.pipeline import Pipeline		
	from sklearn.neural_network import MLPRegressor, MLPClassifier
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.naive_bayes import MultinomialNB
	
	# Path to post and pre processed speechdoc
	spath = '../review_speechdoc_proc'

	regressor = Pipeline([
		('norm', TextNormalizer()),
		('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
		('clf', MLPRegressor(hidden_layer_sizes=[550,150], verbose=True))
	])
	regression_scores = train_model(spath, regressor, continuous=True)
	
	
	classifier = Pipeline([
		('norm', TextNormalizer()),
		('vec', GensimVectorizer()),
		('bayes', MultinomialNB())
	])
	classifier_scores = train_model(spath, classifier, continuour=False)
	

