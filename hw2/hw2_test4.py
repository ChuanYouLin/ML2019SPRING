import pickle
import numpy as np
import pandas as pd
import sys

#读取Model
with open('clf2.pickle', 'rb') as f:
	clf = pickle.load(f)

x = []
f = open(sys.argv[1],"r")
for line in f.readlines():
	x.append(line.split(','))
x.pop(0)
for i in x:
	for j in range(len(i)):
		i[j] = float(i[j])

x_out = np.array(x)

for i in range(1,7):
	if i == 3:
		continue
	max_ = np.max(x_out[:,i],0)
	min_ = np.min(x_out[:,i],0)
	for j in range(x_out.shape[0]):
		x_out[j,i] = (x_out[j,i]-min_) / (max_-min_)

x_out = np.delete(x_out,[8, 13, 39, 64, 78, 79, 80, 81, 88, 91, 97, 100, 101, 104],axis=1)
x_out = np.insert(x_out,0,1.0,axis=1)

#print(x_out.shape)

y_out = clf.predict(x_out)

for i in range(len(y_out)):
	if y_out[i] < 0:
		y_out[i] = 0
	else:
		y_out[i] = 1

y_out = list(y_out)

for i in range(len(y_out)):
	y_out[i] = int(y_out[i])

ans_id = np.arange(1,16282)

dataframe = pd.DataFrame({'id':list(ans_id),'label':y_out})
dataframe.to_csv(sys.argv[2],index=False,sep=',')