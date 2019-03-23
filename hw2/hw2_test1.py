import numpy as np
import math
import sys
import pandas as pd

x = []
f = open(sys.argv[1],"r")
for line in f.readlines():
	x.append(line.split(','))
x.pop(0)
for i in x:
	i.insert(0,1)
	for j in range(len(i)):
		i[j] = float(i[j])

x_in = np.array(x)
for i in range(len(x_in)):
	x_in[i] = x_in[i].T

mean = np.mean(x_in, axis = 0) 
std = np.std(x_in, axis = 0)
for i in range(x_in.shape[0]):
	for j in range(x_in.shape[1]):
		if not std[j] == 0 :
			x_in[i][j] = (x_in[i][j]- mean[j]) / std[j]


w = np.load('w_ver1.npy')

ans = []
for i in range(len(x_in)):
	if w.T.dot(x_in[i]) <= 0:
		ans.append(0)
	else:
		ans.append(1)

ans_id = np.arange(1,16282)

dataframe = pd.DataFrame({'id':list(ans_id),'label':ans})
dataframe.to_csv(sys.argv[2],index=False,sep=',')