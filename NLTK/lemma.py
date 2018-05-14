# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:59:16 2018

@author: shweta
"""

from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", 'a'))
print(lemmatizer.lemmatize("corpora"))