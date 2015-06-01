places = []

myFile = open('../data/articles.csv','w')

with open('../data/places.csv') as f:
	for place in f:
		places.append(place.rstrip()) # remove any new lines ect

with open('../data/raw_articles.csv') as art:
	for a in art:
		tmp = a.split('|')
		
		for p in places:
			if(p in tmp[2]): # If the main text contains a place name
				myFile.write(a) # Write it out to file
				break
		
		
