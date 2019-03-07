#original

import numpy as np
import sys
import csv

def adagrad(x_train,y_train,epoch,lr,size,model):
	best = 1000000000000
	w = np.zeros([size,1])
	w_pre_grad = 0

	w_best = np.zeros([size,1])
	
	for i in range(epoch):
		
		x_train_t = x_train.transpose()

		w_grad = -2*x_train_t.dot(y_train-x_train.dot(w))

		w_pre_grad += w_grad.transpose().dot(w_grad)

		w_ada = w_pre_grad**0.5

		w -= w_grad*lr/w_ada

		ans = x_train.dot(w)
		if ((y_train-ans).transpose().dot(y_train-ans)/y_train.shape[0])**0.5 < best:
			best = min(best,((y_train-ans).transpose().dot(y_train-ans)/y_train.shape[0])**0.5)
			w_best = w.copy()
		if i % 1000 == 0:
			print("epochs = {} loss = {}".format(i,best))
	np.save(model,w_best)



def linear_re(doc,model):
	x = []
	#x_train = np.zeros([18,1])
	with open(doc, newline='',encoding="big5") as csvfile:
		rows = csv.reader(csvfile)
		i = 0
		for row in rows:
			for j in range(len(row)):
				if row[j] == 'NR':
					row[j]=0
			if i == 0:
				i += 1
				continue
			if int((i-1) / 18) == 0:
				x.append(row[3:])
			else:
				for k in row[3:]:
					x[(i-1)%18].append(k)
			#print(row)
			i += 1


	x_train_pre = np.array(x,dtype='float64')
	#print(x_train[-1][:24])
	
	#extract feature
	x_train = []
	y_train = []
	for data_num in range(len(x_train_pre[0])-9):
		c = x_train_pre[:,data_num:data_num+9].copy()
		d = x_train_pre[9,data_num+9]
		
		c_re = np.reshape(c,(c.shape[0]*c.shape[1]))
		c_re = list(c_re)
		c_re.insert(0,1)
		c_re = np.array(c_re)

		x_train.append(c_re)
		y_train.append(d)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	y_train = np.reshape(y_train,(y_train.shape[0],1))
	#print(x_train.shape)
	#print(y_train.shape)

	size = c.shape[0]*c.shape[1]+1
	adagrad(x_train,y_train,100000,10000,size,model)




def main():
	doc = sys.argv[1]
	model = sys.argv[2]
	linear_re(doc,model)

if __name__ == "__main__": main()