#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import getArticle
import copy
import csv
from nltk.tag import pos_tag, map_tag

places = []
another = True

# Load the place names into the program
def loadPlaces():
	with open('../data/places.csv') as place:
		for p in place:
			places.append(p.rstrip())

# See if the sentence contains a place name
def containsPlace(sentence):
	tmp = []
	for p in places:
		if p in sentence:
			tmp.append(p)
	return tmp

# Get the index of the placename
def getPlaceIndex(place, taggedSentence):
	for t in taggedSentence:
		if t[0] == place:
			return taggedSentence.index(t)

# Get the tag of a given word
def labelArray(label):
	
	labels = ['N','N','N','N','N','N','N','N','N']

	if(label == 'VERB'):
		labels[0] = 'Y'
		return labels
	elif(label == 'NOUN'):
		labels[1] = 'Y'
		return labels
	elif(label == 'PRON'):
		labels[2] = 'Y'
		return labels
	elif(label == 'ADJ'):
		labels[3] = 'Y'
		return labels
	elif(label == 'ADV'):
		labels[4] = 'Y'
		return labels
	elif(label == 'ADP'):
		labels[5] = 'Y'
		return labels
	elif(label == 'CONJ'):
		labels[6] = 'Y'
		return labels
	elif(label == 'DET'):
		labels[7] = 'Y'
		return labels
	elif(label == '.'):
		labels[8] = 'Y'
		return labels

	return labels

## Method used to label the class of the place name
def defClass(sentences, s, place):
	sen = copy.deepcopy(sentences)
	index = sen.index(s)
	sen[index] = s.replace(place, '\033[91m' + place + '\033[0m')
	print ''.join(sen)
	answered = ''
	while(not answered):
		ans = raw_input("Is this talking about " + place + "? T/F ")
		if(ans == 'T' or ans == 'F' or ans == 'U'):
			answered = True
	return ans

# Write all of the labelled data out to file
def writeToFile(data):
	with open("../data/labelled_data.csv", "a") as f:
		writer = csv.writer(f)
		writer.writerow(data)

# Add all the automated tags to the place name
def tagWord(place, easyTags, sentences, s):
	index = getPlaceIndex(place, easyTags)

	#==========================================
	# Before
	#==========================================
	if(index == 0):
		before = ['N','N','N','N','N','N','N','N','N']
	else:
		before = labelArray(easyTags[index-1][1])

	#==========================================
	# After
	#==========================================
	if(index == len(easyTags)-1):
		after = ['N','N','N','N','N','N','N','N','N']
	else:
		after = labelArray(easyTags[index+1][1])

	#==========================================
	# at, in, near before
	#==========================================
	aib = ['N','N','N','N','N']
	if(easyTags[index-1][0] == 'at'):
		aib[0] = 'Y'
	elif(easyTags[index-1][0] == 'in'):
		aib[1] = 'Y'
	elif(easyTags[index-1][0] == 'near'):
		aib[2] = 'Y'
	elif(easyTags[index-1][0] == 'the'):
		aib[3] = 'Y'
	elif(easyTags[index-1][0] == 'of'):
		aib[4] = 'Y'

	#==========================================
	# lenghts and numbers (start of sen, end of sen, len of sen and sentence number, word number)
	#==========================================
	lan = ['N','N',len(easyTags),sentences.index(s)+1,str(index+1)] # instead of len(s.split(' '))
	
	if(index == 0):
		lan[0] = 'Y'

	if(index+1 == len(easyTags)-1 and easyTags[index+1][1] == '.'):
		lan[1] = 'Y'
	
	ID = 'S' + str(sentences.index(s)+1) + 'W' + str(index+1)

	data = [article[0], ID]
	data += before + after + aib + lan
	
	data.append(defClass(sentences, s, place))
	writeToFile(data)

## MAIN PROGRAM

loadPlaces() ## LOAD PLACES INTO SYSTEM

while another == True:

	article = getArticle.getArticle()
	
	full_text = article[2]
	sentences = nltk.tokenize.sent_tokenize(full_text)

	for s in sentences:
		place_n = containsPlace(s)
		if(len(place_n) > 0):
			text = nltk.word_tokenize(s)
			posTagged = pos_tag(text)
			easierTagged = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]

			for place in place_n:

				tagWord(place, easierTagged, sentences, s)
 		
	if(raw_input("Would you like to answer another? (y/n) ") != 'y'):
		another = False
