import sys
import csv
import numpy as np
import torch
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

	x_choose = []
	y_choose = []
	k = 0
	for i in range(len(y)):
		if k == 7:
			break
		if int(y[i]) == k:
			x_choose.append(x[i])
			y_choose.append(y[i])
			k += 1

	#normalization to 0~1
	x = np.array(x_choose,dtype=float) / 255.0
	y = np.array(y_choose,dtype=int)
	x = np.reshape(x,(x.shape[0],1,48,48))
	
	'''
	for i in range(2):
		x = np.concatenate((x,x))
		y = np.concatenate((y,y))
	'''
	x = torch.tensor(x)
	y = torch.tensor(y)

	return x,y

def compute_saliency_maps(x, y, model):
	model.eval()
	x.requires_grad_()
	y_pred = model(x.cuda())
	loss_func = torch.nn.CrossEntropyLoss()
	loss = loss_func(y_pred, y.cuda())
	loss.backward()

	saliency = x.grad.abs().squeeze().data
	return saliency

def show_saliency_maps(x, y, model):
	x_org = x.squeeze().numpy()
	# Compute saliency maps for images in X
	saliency = compute_saliency_maps(x, y, model)

	# Convert the saliency map from Torch Tensor to numpy array and show images
	# and saliency maps together.
	saliency = saliency.detach().cpu().numpy()

	num_pics = x_org.shape[0]
	for i in range(num_pics):
		# You need to save as the correct fig names
		#plt.imsave('{}org1_{}.jpg'.format(sys.argv[2],i), x_org[i], cmap=plt.cm.gray)
		plt.imsave('{}fig1_{}.jpg'.format(sys.argv[2],i), saliency[i], cmap=plt.cm.jet)


def main():
	print("Run Hw4_1 ...")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	doc = sys.argv[1]
	x_train,y_train = load_data(doc)
	x_train = x_train.float()

	model = Model()
	model.load_state_dict(torch.load('best.pkl'))
	model = model.to(device)

	show_saliency_maps(x_train, y_train, model)

if __name__ == "__main__": main()