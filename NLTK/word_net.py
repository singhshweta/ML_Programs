# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:59:05 2018

@author: shweta
"""

from nltk.corpus import wordnet

syn_set = wordnet.synsets("program")

#list of synsets
print(syn_set) 

#one synset
print(syn_set[0].name())

#gives just the word
print(syn_set[0].lemmas()[0].name())

#definition
print(syn_set[0].definition())

#examples
print(syn_set[0].examples())
 
#synonyms and antonyms 

synonyms = []
antonyms = []
syn_set = wordnet.synsets("good")
for x in syn_set:
    for l in x.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))  

#similarity between words

word1 = wordnet.synset("airplane.n.01")
word2 = wordnet.synset("helicopter.n.01")

print(word1.wup_similarity(word2)) 

word1 = wordnet.synset("airplane.n.01")
word2 = wordnet.synset("bird.n.01")

print(word1.wup_similarity(word2)) 