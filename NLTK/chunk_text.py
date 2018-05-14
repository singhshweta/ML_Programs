# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:30:57 2018

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
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tag = nltk.pos_tag(words)
            chunkGram = """chunk: {<.*>+}
                                  }<VB.?|IN|DT|TO>{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tag)
            #print(chunked)
            chunked.draw()
#            print(tag)
    except Exception as e:
        print(str(e))

       
process_content()
            
            


