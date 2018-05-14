# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:55:50 2018

@author: shweta
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:15:01 2018

@author: shweta
"""

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize

#PunktSenteceTokenizer:unsupervised machine learning tokenizer
train_text = state_union.raw("2005-GWBush.txt")
text = state_union.raw("2006-GWBush.txt")

sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = sent_tokenizer.tokenize(text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tag = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tag, binary=True)
            namedEnt.draw()
#            print(tag)
    except Exception as e:
        print(str(e))

       
process_content()
            
#binary = true: if we don't care about the name of the entity . It only marks the entities with a common name             


