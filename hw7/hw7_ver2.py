import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import sys
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torchvision.models as models

def load_data(loaded=True):
	if not loaded:
		mypath = "./images/"
		files = sorted(listdir(mypath))
		x = []
		for f in files:
			fullpath = join(mypath,f)
			pic = Image.open(fullpath)
			pic = np.array(pic)
			x.append(pic)
		x = np.array(x)
		np.save("train_x.npy",x)
	else:
		x = np.load("train_x.npy")

	return x[4000:],x[:4000]

class ImageDataset(Dataset):
	def __init__(self, x_train, transform=None):
		self.x_train = x_train
		self.transform = transform
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):

		if self.transform:
			sample_x = self.transform(self.x_train[idx].astype('uint8'))

		return sample_x

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(3),
			nn.ReLU(inplace=True),
		)
	def forward(self, x):
		out = self.encoder(x)
		out = self.decoder(out)
		return out


def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	x_train,x_val = load_data()
	train_dataset = ImageDataset(x_train,
									transforms.Compose([
										#data augmentation
										transforms.ToPILImage(),
										transforms.RandomAffine(degrees=30, translate=(0.2,0.2), scale=(0.8,1.2), shear=0.2),
										transforms.RandomHorizontalFlip(p=0.5),
										transforms.ToTensor(),
										#transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
										]))
	val_dataset = ImageDataset(x_val,
									transforms.Compose([
										#data augmentation
										#transforms.ToPILImage(),
										#transforms.RandomAffine(degrees=60, translate=(0.3,0.3), scale=(0.7,1.3), shear=0.3),
										#transforms.RandomHorizontalFlip(p=0.5),
										transforms.ToTensor(),
										#transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
										]))
	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
	val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)

	model = Model()
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.MSELoss()

	best = 1.0

	for epoch in range(500):
		train_loss = []
		val_loss = []
		model.train()
		for _, (img) in enumerate(train_loader):
			
			img_cuda = img.float().to(device)

			optimizer.zero_grad()
			output = model(img_cuda)

			loss = loss_fn(output, img_cuda)
			loss.backward()
			optimizer.step()

			train_loss.append(loss.item())
		model.eval()
		for _, (img) in enumerate(val_loader):
			
			img_cuda = img.float().to(device)

			output = model(img_cuda)

			loss = loss_fn(output, img_cuda)

			val_loss.append(loss.item())
		print("Epoch: {}, Loss: {:.4f}, Val: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(val_loss)))
		val = np.mean(val_loss)
		if val < best:
			best = val
			print("Model Saved")
			torch.save(model.state_dict(),'ae_ver2.pkl')

if __name__ == '__main__':
	main()
