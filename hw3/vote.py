import numpy as np
import sys
import csv
import torch
import torch.nn as nn
from hw3_ver1 import Model as Model1
from test import Model as Model2
from test2 import Model as Model3
from test3 import Model as Model4
from test4 import Model as Model5
from test5 import Model as Model6
from test6 import Model as Model7
from test7 import Model as Model8
import pandas as pd
import os

def load_data(doc):
	x = []
	y = []
	#x_train = np.zeros([18,1])
	with open(doc, newline='') as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:
			x.append(row[1].split())
	x.pop(0)
	x = np.array(x,dtype=float) / 255.0
	x = np.reshape(x,(x.shape[0],48,48))
	x = torch.tensor(x)
	#print(x.size())
	
	'''z = []
	for i in range(len(x)):
		z.append(np.reshape(x[i],(1,48,48)))
	z = torch.tensor(np.array(z))'''
	return x

def predict(test_dataset,model):
	device = torch.device('cuda')
	model.eval()
	model_cnn = model.to(device)
	test_dataset_cuda = test_dataset.float().to(device)

	label = np.array([0])
	for i in range(48):
		j = (i + 1) * 150
		if j > 7178:
			j = 7178
		test_output = model_cnn(test_dataset_cuda[150*i:j])
		pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
		label = np.concatenate((label, pred_y), axis=0)
		#print(pred_y, 'prediction number')
	label = np.delete(label,0)
	return label

def main():
	print("Start Testing ...")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	doc = sys.argv[1]
	test_dataset = load_data(doc)
	test_dataset = test_dataset.view((7178,1,48,48))
	#print(test_dataset.size())

	#print(test_dataset[0])
	#print(type(test_dataset))
	
	device = torch.device('cuda')

	model = [Model1(),Model2(),Model3(),Model4(),Model5(),Model6(),Model7(),Model8()]

	for i in range(8):
		model[i].load_state_dict(torch.load('best{}.pkl'.format(i)))

	print("Bagging ...")
	label = []
	for i in range(8):
		label.append(predict(test_dataset,model[i]))
		print("model{} is predicting ...".format(i))
		#print(predict(test_dataset,model[i])[14],predict(test_dataset,model[i])[106],predict(test_dataset,model[i])[109])

	result = []
	for i in range(7178):
		count = [0,0,0,0,0,0,0,0]
		for j in range(8):
			if j == 1:
				count[label[j][i]] += 1.5
			else:
				count[label[j][i]] += 1
		result.append(count.index(max(count)))

	#print(result[14])
	print('Create result.csv ...')
	ans_id = np.arange(7178)
	
	dataframe = pd.DataFrame({'id':ans_id,'label':result})
	dataframe.to_csv(sys.argv[2],index=False,sep=',')



if __name__ == "__main__": main()