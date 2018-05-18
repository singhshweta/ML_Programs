# -*- coding: utf-8 -*-
"""
Created on Thu May 17 02:02:34 2018

@author: shweta
"""

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classfiers = classifiers
    
    def classifiy(self,features):
        votes = []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("pickled_algos/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k.pickle","rb")
word_features = pickle.load(word_features5k_f) 
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features
    
featureset_f = open("pickled_algos/featureset.pickle","rb")
featureset = pickle.load(featureset_f)
featureset_f.close()

random.shuffle(featureset)
print(len(featureset))

test_set = featureset[10000:]
train_set = featureset[:10000]

files = open("pickled_algos/naivebayes5k.pickle","rb")
classifier = pickle.load(files)
files.close()

files = open("pickled_algos/MNB_classifier5k.pickle","rb")
MNB_classifier = pickle.load(files)
files.close()

files = open("pickled_algos/BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_classifier = pickle.load(files)
files.close()

files = open("pickled_algos/LogisticRegression_classifier5k.pickle","rb")
LogisticRegression_classifier = pickle.load(files)
files.close()

files = open("pickled_algos/LinearSVC_classifier5k.pickle","rb")
LinearSVC_classifier = pickle.load(files)
files.close()

files = open("pickled_algos/SGD_classifier5k.pickle","rb")
SGD_classifier = pickle.load(files)
files.close()

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
                                 

def sentiment(text):
    feature = find_features(text)
    return voted_classifier.classifiy(feature), voted_classifier.confidence(feature)




