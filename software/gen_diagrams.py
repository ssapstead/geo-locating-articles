import matplotlib.pyplot as plt
import numpy as np

data_kf = []
data_skf = []

#==============================================================================
# Load Data in
#==============================================================================

with open("../output/data_kf.csv", "r") as f:
    for line in f:
		data_kf.append(line.strip().split(','))

with open("../output/data_skf.csv", "r") as f:
    for line in f:
		data_skf.append(line.strip().split(','))

#==============================================================================
# Sort Data into Arrays
#==============================================================================

data_kf_labels = data_kf[0]
data_skf_labels = data_skf[0]

data_kf.remove(data_kf[0])
data_skf.remove(data_skf[0])

tree_kf = []
rm_kf = []
svm_kf = []

for k in data_kf:
	tree_kf.append(float(k[0]))
	rm_kf.append(float(k[1]))
	svm_kf.append(float(k[2]))

tree_skf = []
rm_skf = []
svm_skf = []

for k in data_skf:
	tree_skf.append(float(k[0]))
	rm_skf.append(float(k[1]))
	svm_skf.append(float(k[2]))

#==============================================================================
# Generate Tables
#==============================================================================

f = open('../write_up/tables/kFold.tex','w')
f.write('\\begin{table}' + '\n\\centering' + '\n\\caption{Kold}' + '\n\\label{tb:kFold}' + '\n\\begin{tabular}{|c|c|c|c|} \\hline')
f.write('\nFold&Decision\\_Tree&Random\\_Forest&SVM\\\ \hline\n')
for i in range(len(data_kf)):
	f.write(str(i) + '&' + str(data_kf[i][0]) + '&' + str(data_kf[i][1]) + '&' + str(data_kf[i][2]) + '\\\ \hline \n')
f.write('& & & \\\ \hline \n \\textbf{Average}&' + str(sum(tree_kf)/len(tree_kf)) + '&' + str(sum(rm_kf)/len(rm_kf)) + '&' + str(sum(svm_kf)/len(svm_kf)) + '\\\ \hline \n')
f.write('\\end{tabular}')
f.write('\\end{table}')
f.close()

f = open('../write_up/tables/stratified_kFold.tex','w')
f.write('\\begin{table}' + '\n\\centering' + '\n\\caption{Stratified Kold}' + '\n\\label{tb:st_kFold}' + '\n\\begin{tabular}{|c|c|c|c|} \\hline')
f.write('\nFold&Decision\\_Tree&Random\\_Forest&SVM\\\ \hline\n')
for i in range(len(data_skf)):
	f.write(str(i) + '&' + str(data_skf[i][0]) + '&' + str(data_skf[i][1]) + '&' + str(data_skf[i][2]) + '\\\ \hline \n')
f.write('& & & \\\ \hline \n \\textbf{Average}&' + str(sum(tree_skf)/len(tree_skf)) + '&' + str(sum(rm_skf)/len(rm_skf)) + '&' + str(sum(svm_skf)/len(svm_skf)) + '\\\ \hline \n')
f.write('\\end{tabular}')
f.write('\\end{table}')
f.close()

###############################################################################

f = open('../write_up/tables/kf_comp.tex','w')
f.write('\\begin{table}' + '\n\\centering' + '\n\\caption{KFold Comparison}' + '\n\\label{tb:kf_compare}' + '\n\\begin{tabular}{|c|c|c|c|} \\hline')
f.write('\n &Decision\\_Tree&Random\\_Forest&SVM\\\ \hline\n')

f.write("Biggest &" + str(max(tree_kf)) + '&' + str(max(rm_kf)) + '&' + str(max(svm_kf)) + '\\\ \hline \n')
f.write("Smallest &" + str(min(tree_kf)) + '&' + str(min(rm_kf)) + '&' + str(min(svm_kf)) + '\\\ \hline \n')
f.write("Range &" + str(max(tree_kf)-min(tree_kf)) + '&' + str(max(rm_kf)-min(rm_kf)) + '&' + str(max(svm_kf)-min(svm_kf)) + '\\\ \hline \n')
f.write("Average &" + str(sum(tree_kf)/len(tree_kf)) + '&' + str(sum(rm_kf)/len(rm_kf)) + '&' + str(sum(svm_kf)/len(svm_kf)) + '\\\ \hline \n')

f.write('\\end{tabular}')
f.write('\\end{table}')
f.close()

########

f = open('../write_up/tables/skf_comp.tex','w')
f.write('\\begin{table}' + '\n\\centering' + '\n\\caption{Stratified KFold Comparison}' + '\n\\label{tb:skf_compare}' + '\n\\begin{tabular}{|c|c|c|c|} \\hline')
f.write('\n &Decision\\_Tree&Random\\_Forest&SVM\\\ \hline\n')

f.write("Biggest &" + str(max(tree_skf)) + '&' + str(max(rm_skf)) + '&' + str(max(svm_skf)) + '\\\ \hline \n')
f.write("Smallest &" + str(min(tree_skf)) + '&' + str(min(rm_skf)) + '&' + str(min(svm_skf)) + '\\\ \hline \n')
f.write("Range &" + str(max(tree_skf)-min(tree_skf)) + '&' + str(max(rm_skf)-min(rm_skf)) + '&' + str(max(svm_skf)-min(svm_skf)) + '\\\ \hline \n')
f.write("Average &" + str(sum(tree_skf)/len(tree_skf)) + '&' + str(sum(rm_skf)/len(rm_skf)) + '&' + str(sum(svm_skf)/len(svm_skf)) + '\\\ \hline \n')

f.write('\\end{tabular}')
f.write('\\end{table}')
f.close()


#==============================================================================
# Generate Single KFold Graphs
#==============================================================================

axis = [0,1,2,3,4,5,6,7,8,9]

plt.plot(axis, tree_kf, 'b', label='Accuracy')
av = sum(tree_kf)/len(tree_kf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Decision Tree Accuracy (KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/dt_single_kf.png', dpi=400)
plt.close()

plt.plot(axis, rm_kf, 'b', label='Accuracy')
av = sum(rm_kf)/len(rm_kf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Random Forest Accuracy (KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/rf_single_kf.png', dpi=400)
plt.close()

plt.plot(axis, svm_kf, 'b', label='Accuracy')
av = sum(svm_kf)/len(svm_kf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Support Vector Machine Accuracy (KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/svm_single_kf.png', dpi=400)
plt.close()

#==============================================================================
# Generate Single Stratified KFold Graphs
#==============================================================================

axis = [0,1,2,3,4,5,6,7,8,9]

plt.plot(axis, tree_skf, 'b', label='Accuracy')
av = sum(tree_skf)/len(tree_skf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Decision Tree Accuracy (Stratified KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/dt_single_skf.png', dpi=400)
plt.close()

plt.plot(axis, rm_skf, 'b', label='Accuracy')
av = sum(rm_skf)/len(rm_skf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Random Forest Accuracy (Stratified KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/rf_single_skf.png', dpi=400)
plt.close()

plt.plot(axis, svm_skf, 'b', label='Accuracy')
av = sum(svm_skf)/len(svm_skf)
av = [av]*10
plt.plot(axis, av, 'g--', label='Average')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Support Vector Machine Accuracy (Stratified KFold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/svm_single_skf.png', dpi=400)
plt.close()

#==============================================================================
# Generate Comparasen KFold Graphs
#==============================================================================

axis = [0,1,2,3,4,5,6,7,8,9]

plt.plot(axis, tree_skf, 'b', label='SKFold Accuracy')
plt.plot(axis, tree_kf, 'g', label='KFold Accuracy')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Decision Tree Accuracy Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/dt_comp.png', dpi=400)
plt.close()

plt.plot(axis, rm_skf, 'b', label='SKFold Accuracy')
plt.plot(axis, rm_kf, 'g', label='KFold Accuracy')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Random Forest Accuracy Comparision")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/rf_comp.png', dpi=400)
plt.close()

plt.plot(axis, svm_skf, 'b', label='SKFold Accuracy')
plt.plot(axis, svm_kf, 'g', label='KFold Accuracy')
legend = plt.legend(loc='upper right')
plt.ylim([0.5,1.0])
plt.title("Support Vector Machine Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/svm_comp.png', dpi=400)
plt.close()


#==============================================================================
# All Three KFold Graphs
#==============================================================================

plt.plot(axis, tree_skf, 'b', label='Tree Accuracy')
plt.plot(axis, rm_kf, 'g', label='RF Accuracy')
plt.plot(axis, svm_kf, 'r', label='SVM Accuracy')
legend = plt.legend(loc='upper left')
plt.ylim([0.5,1.0])
plt.title("KFold Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/kf_comp.png', dpi=400)
plt.close()

#==============================================================================
# All Three Strat KFold Graphs
#==============================================================================

plt.plot(axis, tree_skf, 'b', label='Tree Accuracy')
plt.plot(axis, rm_skf, 'g', label='RF Accuracy')
plt.plot(axis, svm_skf, 'r', label='SVM Accuracy')
legend = plt.legend(loc='upper left')
plt.ylim([0.5,1.0])
plt.title("Stratified KFold Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.savefig('../write_up/diagrams/skf_comp.png', dpi=400)
plt.close()
