import sys
import csv
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from model.test import Model
import os
import torch.nn as nn
import random


class Model2(nn.Module):
	def __init__(self):
		super(Model2, self).__init__()
		self.conv2 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=5),
			nn.LeakyReLU(0.05,inplace=True),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.1),
			nn.Conv2d(64, 128, kernel_size=3),
			nn.LeakyReLU(0.05,inplace=True),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.2),
			nn.Conv2d(128, 512, kernel_size=3),
			nn.LeakyReLU(0.05,inplace=True),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.25),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(0.05,inplace=True),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3),
			nn.LeakyReLU(0.05,inplace=True),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.3),
		)

	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.conv2(x)
		out = out.view(512,1,1)
		ret = torch.tensor(np.zeros(512))
		for i in range(out.shape[0]):
			ret[i] = out[i].sum()
		#print(ret.shape)
		return ret


def get_features_hook(self, input, output):
	print("hook",output.data.cpu().numpy().shape)
	out = output.data.cpu().numpy().squeeze()




def main():
	print("Run Hw4_2_1 ...")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda')

	model = Model()
	model.eval()
	model.load_state_dict(torch.load('best.pkl'))
	model = model.to(device)
	model.conv2[0].register_forward_hook(get_features_hook)

	model2 = Model2()
	model2 = model2.to(device)

	pretrained_dict = model.state_dict()    
	model_dict = model2.state_dict()
	pretrained_dicted = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dicted)
	model2.load_state_dict(model_dict)

	'''for k, v in model2.state_dict().items():
		print("Layer {}".format(k))
	print(model.state_dict()['conv2.0.weight'].shape)'''
	torch.manual_seed(87)
	random.seed(87)
	for f in range(8):
		x = torch.rand(1,1,48,48)

		x.requires_grad_()
		x_cuda = x.float().to(device)

		for i in range(500):
			z = model2(x_cuda)[f]
			z.backward()
			#print(x.grad)
			y = x.grad
			#y = torch.tensor(np.pad(y,((0,0),(0,0),(2,2),(2,2)),'constant',constant_values = (0.,0.)))
			#print(y.shape)
			x_cuda += (0.01*y).float().to(device)
			#print(x_cuda)
		plt.subplot(2,4,f+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(x_cuda.detach().cpu().numpy().squeeze().squeeze(), cmap=plt.cm.gray)

	plt.savefig('{}fig2_1.jpg'.format(sys.argv[2]))
	

if __name__ == "__main__": main()