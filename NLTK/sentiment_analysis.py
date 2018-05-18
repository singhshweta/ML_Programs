# -*- coding: utf-8 -*-
"""
Created on Thu May 17 01:16:31 2018

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
        
short_pos = open("short_reviews/positive.txt","r",encoding='latin-1').read()
short_neg = open("short_reviews/negative.txt","r",encoding='latin-1').read()

all_words = []
documents = []

#j is adjective , r is adverb, v is verb
#allowed_word_types = ["J","R","V"]

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if(w[1][0]) in allowed_word_types:
            all_words.append(w[0].lower())
            
for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if(w[1][0]) in allowed_word_types:
            all_words.append(w[0].lower())
            
save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features,save_features)
save_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features
    
featureset = [(find_features(rev),category) for (rev,category) in documents]

save_featureset = open("pickled_algos/featureset.pickle","wb")
pickle.dump(featureset,save_featureset)
save_featureset.close()

random.shuffle(featureset)
print(len(featureset))

test_set = featureset[10000:]
train_set = featureset[:10000]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("naive bayes accuracy percent:", (nltk.classify.accuracy(classifier,test_set))*100)


save_classifier  = open("pickled_algos/naivebayes5k.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,test_set))*100)

save_classifier  = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier,test_set))*100)

save_classifier  = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,test_set))*100)

save_classifier  = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,test_set))*100)

save_classifier  = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()

#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(train_set)
#print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,test_set))*100)


SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train_set)
print("SGD_classifier accuracy percent:", (nltk.classify.accuracy(SGD_classifier,test_set))*100)

save_classifier  = open("pickled_algos/SGD_classifier5k.pickle","wb")
pickle.dump(SGD_classifier,save_classifier)
save_classifier.close()

