import xml.etree.ElementTree as ET
tree = ET.parse('../data/news_1900.xml')
root = tree.getroot()

with open('../data/raw_articles.csv', 'w') as f:
	for child in root:
		print child[11].text
		data = ''.join(child[11].text + '|' + child[12].text + '|' + child[22].text + '\n').encode('utf-8')
		f.write(data)

