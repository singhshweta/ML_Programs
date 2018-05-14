# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:27:21 2018

@author: shweta
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = []
for  category in movie_reviews.categories():
    for fileid in movie_reviews.fileids():
        documents.append((list(movie_reviews.words(fileid)),category))

#print(documents[5:20])
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(20))
#print(all_words["boring"])

word_features = list(all_words.keys()[:3000])

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = w in document
    return features
    
#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev),category) for (rev, category) in documents]

#print (featuresets)

train_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("acuuracy percent", (nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(20)

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier , save_classifier)
save_classifier.close()

#how to load the saved classifier

#classifier_f = open("naivebayes.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()
