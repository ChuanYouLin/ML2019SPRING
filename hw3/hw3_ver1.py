import numpy as np
import sys
import csv
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
	x = np.array(x,dtype=int)
	y = np.array(y,dtype=int)
	x = np.reshape(x,(x.shape[0],48,48))
	
	'''
	for i in range(2):
		x = np.concatenate((x,x))
		y = np.concatenate((y,y))
	'''
	np.random.seed(87)
	randomize = np.arange(len(x))
	np.random.shuffle(randomize)
	x , y = x[randomize], y[randomize]

	return x[:-2000],y[:-2000],x[-2000:],y[-2000:]


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv2 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, padding=2),
			nn.LeakyReLU(0.05),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.2),
			nn.Conv2d(32, 256, kernel_size=3, padding=1),
			nn.LeakyReLU(0.05),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.3),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(0.05),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.3),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(0.05),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2, 2),
			#nn.Dropout2d(0.3),
		)
		self.fc = nn.Sequential(
			nn.Linear(3*3*512, 512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			#nn.Dropout(0.5),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			#nn.Dropout(0.5),
			nn.Linear(512,7),
		)
		self.output = nn.Softmax(dim=1)

	def forward(self, x):
		# You can modify your model connection whatever you like
		out = self.conv2(x)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out = self.output(out)
		return out


class ImageDataset(Dataset):
	def __init__(self, x_train, y_train, transform=None):
		self.x_train = x_train
		self.y_train = y_train
		self.transform = transform
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):

		if self.transform:
			sample_x = self.transform(self.x_train[idx].astype('uint8'))

		sample_y = torch.tensor(self.y_train[idx])

		return [sample_x,sample_y]


def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	doc = sys.argv[1]
	x_train,y_train,x_val,y_val = load_data(doc)
	
	train_dataset = ImageDataset(x_train,y_train,
									transforms.Compose([
										#data augmentation
										transforms.ToPILImage(),
										transforms.RandomAffine(degrees=30, translate=(0.2,0.2), scale=(0.8,1.2), shear=0.2),
										transforms.RandomHorizontalFlip(p=0.5),
										transforms.ToTensor(),
										]))

	val_dataset = ImageDataset(x_val,y_val,
									transforms.Compose([
										#data augmentation
										transforms.ToTensor(),
										]))
	

	x_val_tensor = val_dataset[0][0]
	y_val_tensor = val_dataset[0][1].reshape(1)
	
	for i in range(1,len(val_dataset)):
		x_val_tensor = torch.cat((x_val_tensor,val_dataset[i][0]),0)
		y_val_tensor = torch.cat((y_val_tensor,val_dataset[i][1].reshape(1)),0)
	
	x_val_cuda = x_val_tensor.view((2000,1,48,48)).float().to(device)
	y_val_cuda = y_val_tensor
	#print(x_val_tensor)
	#print(x_val_tensor.size())
	#print(y_val_tensor.size())
	
	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
	print(len(train_dataset))

	device = torch.device('cuda')
	
	model = Model()
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.CrossEntropyLoss()

	model.train()
	min_val = 0.0

	for epoch in range(1):
		train_loss = []
		train_acc = []
		
		for _, (img, target) in enumerate(train_loader):
			
			img_cuda = img.float().to(device)
			target_cuda = target.to(device)


			optimizer.zero_grad()
			output = model(img_cuda)

			#regularization
			l2_reg = torch.tensor(0.).to(device)
			for param in model.parameters():
				l2_reg += torch.norm(param)**2


			loss = loss_fn(output, target_cuda)
			loss.backward()
			optimizer.step()


			predict = torch.max(output, 1)[1]
			acc = np.mean((target_cuda == predict).cpu().numpy())

			train_acc.append(acc)
			train_loss.append(loss.item())

		label = np.array([0])
		for i in range(4):
			j = (i + 1) * 500
			val_output = model(x_val_cuda[i*500:j])
			pred_y = torch.max(val_output, 1)[1].data.cpu().numpy().squeeze()
			label = np.concatenate((label, pred_y), axis=0)
		label = np.delete(label,0)
		acc_val = np.mean((np.array(y_val_cuda) == label))
		
		print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Validation : {}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc), acc_val))

		if epoch > 100 and acc_val > min_val:
			min_val = acc_val
			torch.save(model.state_dict(),'best0.pkl')
			f = open('record0.txt','a')
			f.write("Epoch: {}, Validation : {}\n".format(epoch + 1, min_val))
	
	#torch.save(model.state_dict(), 'cnn.pkl')

	
if __name__ == "__main__": main()
