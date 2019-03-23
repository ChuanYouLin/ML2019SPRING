#SVM

import numpy as np
import math
import sys
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
import pickle

x = []
f = open("X_train","r")
for line in f.readlines():
	x.append(line.split(','))
x.pop(0)
for i in x:
	for j in range(len(i)):
		i[j] = float(i[j])

y = []
f = open("Y_train","r")
for line in f.readlines():
	y.append(line.split()[0])
y.pop(0)
for i in range(len(y)):
	y[i] = float(y[i])
	if y[i] == 0.0:
		y[i] = -1.0

x_in = np.array(x)
y_in = np.array(y)

for i in range(1,7):
	if i == 3:
		continue
	max_ = np.max(x_in[:,i],0)
	min_ = np.min(x_in[:,i],0)
	for j in range(x_in.shape[0]):
		x_in[j,i] = (x_in[j,i]-min_) / (max_-min_)

sel = VarianceThreshold(threshold=(.9993 * (1 - .9993)))
sel.fit_transform(x_in)
mark = []
for i in range(len(x_in[0])):
	if i not in sel.get_support(1):
		mark.append(i)
x_in = sel.fit_transform(x_in)

x_in = np.insert(x_in,0,1.0,axis=1)
print(mark)
print(x_in.shape)

gamma = [0.001]
C = [10]
par = []
for i in range(len(gamma)):
	for j in range(len(C)):
		par.append([gamma[i],C[j]])

model = []

for i in range(len(par)):
	clf = svm.SVC(gamma=par[i][0], C=par[i][1])

	print("Start SVM : (gamma={}, C={})".format(par[i][0],par[i][1]))
	
	#clf.fit(x_in,y_in)

	scores = cross_val_score(clf, x_in, y_in, cv=5)
	print("Accuracy: %0.5f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

	model.append([scores.mean(),clf,par[i][0],par[i][1]])

model.sort()
model[-1][1].fit(x_in,y_in)
print("Best Accuracy: {}, gamma = {}, C = {}".format(model[-1][0],model[-1][2],model[-1][3]))

with open('clf2.pickle', 'wb') as f:
	pickle.dump(model[-1][1], f)