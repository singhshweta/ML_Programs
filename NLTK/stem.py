# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:56:59 2018

@author: shweta
"""

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

P_S = PorterStemmer()
#words_list = ["development","developer","developing","developed","develops"]

#for w in words_list:
    #print(P_S.stem(w))

text = "I am a website developer. I love doing web development. I have developed two websites and a blog. Web developer develops websites blogs etc."
words = word_tokenize(text)

for i in words :
    print(P_S.stem(i))