# -*- coding: utf-8 -*-
import sys
reload(sys) # needed to be able to set encoding below
sys.setdefaultencoding('Cp1252') ## Had to set the encoding to a windows encoding to stop encoding errors while extracting articles.

doneList = []

## Add all the already labelled articles to a list
with open('../data/done.csv') as done:
	for d in done:
		doneList.append(d.rstrip())

# Get the next article
def getArticle():

	with open('../data/articles.csv') as articles:
		for a in articles:
			a = a.split('|')
			if(a[0] not in doneList):
				addDone(a[0])
				return a

# Add labelled articles to the done file when done
def addDone(ID):
	doneList.append(ID) ## Would an update of doneList be better?
	with open('../data/done.csv','a') as done:
		done.write(ID + '\n')
