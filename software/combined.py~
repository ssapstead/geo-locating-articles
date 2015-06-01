import numpy as np
import sys

from sklearn import tree, cross_validation, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pylab as pl

dataX = []
dataY = []

if len(sys.argv) > 1:
	dataFile = sys.argv[1]
else:
	dataFile = "../data/converted_data_labelled.csv"

print "USING: ", dataFile

with open(dataFile) as f:
	for d in f:
		d = d.strip().split(',')
		dataX.append(d[2:len(d)-1])
		dataY.append(d[len(d)-1])

labels = dataX[0]

dataX.remove(dataX[0]) # remove headings from csv file
dataY.remove(dataY[0])

dataX = np.array(dataX)
dataY = np.array(dataY)

#====================================================================================================================================================
# KFOLD
#====================================================================================================================================================

print len(dataX)

kf = cross_validation.KFold(len(dataX), n_folds=10)

kf_scores = []

kf_tree_matrix = []
kf_rf_matrix = []
kf_svm_matrix = []

kf_data_out = []

for train_index, test_index in kf:

	X_train, X_test = dataX[train_index], dataX[test_index]
	y_train, y_test = dataY[train_index], dataY[test_index]

	kf_data_out.append([list(y_train).count('0'), list(y_train).count('1'), list(y_test).count('0'), list(y_test).count('1')])
	
	## Tree Classifier
	tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=10, max_depth=8)
	tree_classifier = tree_classifier.fit(X_train, y_train)
	tree_array = tree_classifier.predict(X_test)
	kf_tree_matrix.append(confusion_matrix(y_test, tree_array))
	tree_score = tree_classifier.score(X_test, y_test)

	## Random Forest Classifier
	rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=8)
	rf_classifier = rf_classifier.fit(X_train, y_train)
	rf_array = rf_classifier.predict(X_test)
	kf_rf_matrix.append(confusion_matrix(y_test, rf_array))
	rf_score = rf_classifier.score(X_test, y_test)

	## SVM Classifier
	svm_classifier = svm.SVC()
	svm_classifier = svm_classifier.fit(X_train, y_train)
	svm_array = svm_classifier.predict(X_test)
	kf_svm_matrix.append(confusion_matrix(y_test, svm_array))
	svm_score = svm_classifier.score(X_test, y_test)	

	kf_scores.append([tree_score, rf_score, svm_score])
	

f = open("../output/data_out_kf.csv", "w")
f.write("train0, train1, test0, test1\n")
for s in kf_data_out:
	f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + ',' + str(s[3]) +'\n')
	print str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + ',' + str(s[3])
f.close()


#====================================================================================================================================================
# STRATIFIED KFOLD
#====================================================================================================================================================

skf = cross_validation.StratifiedKFold(dataY, n_folds=10)

skf_scores = []

skf_tree_matrix = []
skf_rf_matrix = []
skf_svm_matrix = []

skf_data_out = []

for train_index, test_index in skf:

	X_train, X_test = dataX[train_index], dataX[test_index]
	y_train, y_test = dataY[train_index], dataY[test_index]
	
	skf_data_out.append([list(y_train).count('0'), list(y_train).count('1'), list(y_test).count('0'), list(y_test).count('1')])

	## Tree Classifier
	tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=10, max_depth=8)
	tree_classifier = tree_classifier.fit(X_train, y_train)
	tree_array = tree_classifier.predict(X_test)
	skf_tree_matrix.append(confusion_matrix(y_test, tree_array))
	tree_score = tree_classifier.score(X_test, y_test)

	## Random Forest Classifier
	rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=8)
	rf_classifier = rf_classifier.fit(X_train, y_train)
	rf_array = rf_classifier.predict(X_test)
	skf_rf_matrix.append(confusion_matrix(y_test, rf_array))
	rf_score = rf_classifier.score(X_test, y_test)

	## SVM Classifier
	svm_classifier = svm.SVC()
	svm_classifier = svm_classifier.fit(X_train, y_train)
	svm_array = svm_classifier.predict(X_test)
	skf_svm_matrix.append(confusion_matrix(y_test, svm_array))
	svm_score = svm_classifier.score(X_test, y_test)	

	skf_scores.append([tree_score, rf_score, svm_score])
	
print "--"
f = open("../output/data_out_skf.csv", "w")
f.write("train0, train1, test0, test1\n")
for s in skf_data_out:
	f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + ',' + str(s[3]) +'\n')
	print str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + ',' + str(s[3])
f.close()

#====================================================================================================================================================
# Average Metrics
#====================================================================================================================================================

def averageMetrics(matrix):
	tl = 0 # top left
	tr = 0 # top right
	bl = 0 # bottom left
	br = 0 # bottom right
	for m in matrix:
		tl += m[0][0]
		tr += m[0][1]
		bl += m[1][0]
		br += m[1][1]

	return [[tl, tr],[bl, br]]


def genMat(data, name):

	return str(name) + ',' + str(data[0][0]) + ',' + str(data[0][1]) + ',' + str(data[1][0]) + ',' + str(data[1][1]) + '\n'

f = open("../output/Matrix Data","w")
f.write(genMat(averageMetrics(kf_tree_matrix), "Decision_Tree_KFold"))
f.write(genMat(averageMetrics(kf_rf_matrix), "Random_Forest_KFold"))
f.write(genMat(averageMetrics(kf_svm_matrix), "SVM_KFold"))

f.write(genMat(averageMetrics(skf_tree_matrix), "Decision_Tree_Stratified_KFold"))
f.write(genMat(averageMetrics(skf_rf_matrix), "Random_Forest_Stratified_KFold"))
f.write(genMat(averageMetrics(skf_svm_matrix), "SVM_Stratified_KFold"))
f.close()
#====================================================================================================================================================
f = open("../output/data_kf.csv", "w")
f.write("Tree, Random Forest, SVM\n")
for s in kf_scores:
	f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + '\n')
f.close()

f = open("../output/data_skf.csv", "w")
f.write("Tree, Random Forest, SVM\n")
for s in skf_scores:
	f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(s[2]) + '\n')
f.close()
