# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:43:23 2018

@author: shweta
"""

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

token = sent_tokenize(sample)

print(token[5:10])