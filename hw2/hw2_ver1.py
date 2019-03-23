#LG

import numpy as np

import math
import sys

def sigmoid(s):
	s = np.clip(s,-500,500)

	s = 1.0/(1+np.exp(-s))

	return s

x = []
f = open("X_train","r")
for line in f.readlines():
	x.append(line.split(','))
x.pop(0)
for i in x:
	i.insert(0,1)
	for j in range(len(i)):
		i[j] = float(i[j])

y = []
f = open("Y_train","r")
for line in f.readlines():
	y.append(line.split())
y.pop(0)
for i in y:
	for j in range(len(i)):
		i[j] = float(i[j])
		if i[j] == 0.0:
			i[j] = -1.0

x_in = np.array(x)
y_in = np.array(y)
for i in range(len(x_in)):
	x_in[i] = x_in[i].T

mean = np.mean(x_in, axis = 0) 
std = np.std(x_in, axis = 0)
for i in range(x_in.shape[0]):
	for j in range(x_in.shape[1]):
		if not std[j] == 0 :
			x_in[i][j] = (x_in[i][j]- mean[j]) / std[j]

w = np.zeros(x_in[0].shape)

E_in_gd = []
for t in range(500):
	eta=1
	err=0
	err_gradient = np.zeros(x_in[0].shape)
	for i in range(len(x_in)):
		#計算Ein梯度
		s = -1*y_in[i]*(w.T).dot(x_in[i])
		s = sigmoid(s)
		r = -1*y_in[i]*(x_in[i])
		err_gradient+=s*r
	err_gradient/=len(x_in)
	w-=eta*err_gradient
	for i in range(len(x_in)):
		s = w.T.dot(x_in[i])
		if s*y_in[i]<0:
			err+=1
	if t >= 0:
		E_in_gd.append(err/len(x_in))
	if t % 10 == 0:
		print(t,err/len(x_in))

np.save('w_ver1.npy',w)
'''E_in_sgd = []
for t in range(5000):
	eta=0.01
	err=0
	err_gradient = np.zeros(x_in[0].shape)
	#計算Ein梯度
	s = -1*y_in[i]*(w.T).dot(x_in[i])
	s = sigmoid(s)
	r = y_in[i]*(x_in[i])
	err_gradient+=s*r
	w+=eta*err_gradient
	i+=1
	if i>=len(x_in):
		i=0
	for k in range(len(x_in)):
		s = w.T.dot(x_in[k])
		if s*y_in[k]<0:
			err+=1
	E_in_sgd.append(err/len(x_in))
	if t%10 == 0:
		print(t,err/len(x_in))'''

