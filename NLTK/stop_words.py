# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:14:21 2018

@author: shweta
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a peak time of summer here. The sun is burning like anything."
stop_words = set(stopwords.words("english"))

words = word_tokenize(text)

filtered_text = []
for i in words :
    if i not in stop_words:
        filtered_text.append(i)

print(filtered_text)

