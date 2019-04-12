import sys
import csv
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from model.test import Model
import os


def load_data(doc):
	x = []
	y = []
	#x_train = np.zeros([18,1])
	with open(doc, newline='') as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:
			y.append(row[0])
			x.append(row[1].split())
	x.pop(0)
	y.pop(0)

	#normalization to 0~1
	x = np.array(x,dtype=float) / 255.0
	y = np.array(y,dtype=int)
	x = np.reshape(x,(x.shape[0],1,48,48))
	
	'''
	for i in range(2):
		x = np.concatenate((x,x))
		y = np.concatenate((y,y))
	'''
	x = torch.tensor(x)
	y = torch.tensor(y)

	return x,y

def get_features_hook(self, input, output):
	#print("hook",output.data.cpu().numpy().shape)
	out = output.data.cpu().numpy().squeeze()
	num_pics = out.shape[0]
	for i in range(num_pics):
		plt.subplot(8,8,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(out[i], cmap=plt.cm.jet)
	plt.savefig('{}fig2_2.jpg'.format(sys.argv[2]))

def main():
	print("Run Hw4_2_2 ...")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda')
	doc = sys.argv[1]
	x_train,y_train = load_data(doc)
	x_train = x_train.float()

	model = Model()
	model.eval()
	model.load_state_dict(torch.load('best.pkl'))
	model = model.to(device)
	model.conv2[0].register_forward_hook(get_features_hook)

	y_pred = model(x_train[0].view(1,1,48,48).to(device))
	

if __name__ == "__main__": main()