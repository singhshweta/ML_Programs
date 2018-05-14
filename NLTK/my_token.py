# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:48:35 2018

@author: shweta
"""

from nltk.tokenize import sent_tokenize, word_tokenize
text = "hello Mr. Khanna,how are you? I am doing good here. The weather is very hot here."
for i in word_tokenize(text):
    print (i)
for i in sent_tokenize(text):
    print(i)
    
