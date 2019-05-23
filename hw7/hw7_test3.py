import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from hw7_ver2 import Model, ImageDataset
import pandas as pd
import os
from sklearn import cluster
import sklearn

def load_data(loaded=False):
	if not loaded:
		mypath = sys.argv[1]
		files = sorted(listdir(mypath))
		x = []
		for f in files:
			fullpath = join(mypath,f)
			pic = Image.open(fullpath)
			pic = np.array(pic)
			x.append(pic)
		x_image = np.array(x)
		#np.save("train_x.npy",x)
	else:
		x_image = np.load("train_x.npy")
	
	x = []
	with open(sys.argv[2], newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			x.append([int(row['image1_name']),int(row['image2_name'])])

	return x_image,x

activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	x_test,test_case = load_data()
	test_dataset = ImageDataset(x_test,
									transforms.Compose([
										#data augmentation
										#transforms.ToPILImage(),
										#transforms.RandomAffine(degrees=30, translate=(0.2,0.2), scale=(0.8,1.2), shear=0.2),
										#transforms.RandomHorizontalFlip(p=0.5),
										transforms.ToTensor(),
										]))
	test_loader = DataLoader(dataset=test_dataset,batch_size=128,shuffle=False)

	model = Model()
	model.to(device)
	model.load_state_dict(torch.load('ae_ver2.pkl'))
	model.eval()
	result = []
	'''test = model(test_dataset[-1].unsqueeze(0).float().to(device))
	test = test.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255
	test_img = Image.fromarray(test.astype('uint8'))
	test_img.save("test.jpg")
	exit(0)'''
	for _, (img) in enumerate(test_loader):
		img = img.float().to(device)
		model.encoder.register_forward_hook(get_activation('encoder'))
		output = model(img)
		output = activation['encoder']

		if _ == 0:
			low = output.view((output.size(0),-1)).detach().cpu().numpy()
		else:
			out = output.view((output.size(0),-1)).detach().cpu().numpy()
			low = np.concatenate((low,out),axis=0)
	#8192
	pca = sklearn.decomposition.PCA(n_components=1024, whiten=True, random_state=9487).fit_transform(low)
	#print(pca.shape)

	kmeans_fit = cluster.KMeans(n_clusters = 2, random_state=9487).fit(pca)
	cluster_labels = kmeans_fit.labels_
	
	#print(cluster_labels[:20])

	'''count = 0
	hey = [0,1,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1]
	for z in range(50):
		if cluster_labels[z] != hey[z]:
			count += 1
	print(count)
	exit(0)'''

	for i in range(len(test_case)):
		if cluster_labels[test_case[i][0]-1] == cluster_labels[test_case[i][1]-1]:
			result.append(1)
		else:
			result.append(0)
	#print(sum(result))
	#exit(0)

	ans_id = np.arange(1000000)
	dataframe = pd.DataFrame({'id':ans_id,'label':result})
	dataframe.to_csv(sys.argv[3],index=False,sep=',')

	# X_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(pca)
	# plt.figure()
	# plt.scatter(X_embedded[:,0],X_embedded[:,1],c=cluster_labels, s=0.5, alpha = 0.5)
	# plt.savefig('gg2.jpg')

if __name__ == '__main__':
	main()
