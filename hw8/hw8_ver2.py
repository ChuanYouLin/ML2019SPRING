import numpy as np
import sys
import csv
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch.nn.functional as F

def load_data(doc):
	x = []
	y = []
	with open(doc, newline='') as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:
			y.append(row[0])
			x.append(row[1].split())
	x.pop(0)
	y.pop(0)

	x = np.array(x,dtype='uint8')
	y = np.array(y,dtype=int)
	x = np.reshape(x,(x.shape[0],48,48))

	np.random.seed(87)
	randomize = np.arange(len(x))
	np.random.shuffle(randomize)
	x , y = x[randomize], y[randomize]

	return x[:-3000],y[:-3000],x[-3000:],y[-3000:]

class ImageDataset(Dataset):
	def __init__(self, x_train, y_train=None, transform=None):
		self.x_train = x_train
		self.y_train = y_train
		self.transform = transform
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):

		if self.transform:
			sample_x = self.transform(self.x_train[idx])
		sample_y = torch.tensor(self.y_train[idx])

		return [sample_x,sample_y]

class Block(nn.Module):
	def __init__(self, in_planes, out_planes, stride=2):
		super(Block, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		return out


class MobileNet(nn.Module):
	cfg = [(32,2), 32, 32, (64,2), 64, 64, (64,2), 64, 64, 64, (128,2)]

	def __init__(self, num_classes=7):
		super(MobileNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layers = self._make_layers(in_planes=16)
		self.linear = nn.Linear(128, num_classes)
		self.output = nn.Softmax(dim=1)

	def _make_layers(self, in_planes):
		layers = []
		for x in self.cfg:
			out_planes = x if isinstance(x, int) else x[0]
			stride = 1 if isinstance(x, int) else x[1]
			layers.append(Block(in_planes, out_planes, stride))
			in_planes = out_planes
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layers(out)
		out = F.avg_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		out = self.output(out)
		return out

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	
	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
	val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)

	model = MobileNet()
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.CrossEntropyLoss()

	min_val = 0.0

	for epoch in range(5000):
		train_loss = []
		train_acc = []
		val_acc = []
		model.train()
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
		model.eval()
		for _, (img, target) in enumerate(val_loader):
			img_cuda = img.float().to(device)
			target_cuda = target.to(device)

			output = model(img_cuda)

			predict = torch.max(output, 1)[1]
			acc = np.mean((target_cuda == predict).cpu().numpy())

			val_acc.append(acc)

		print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Validation : {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc), np.mean(val_acc)))

		if np.mean(val_acc) > min_val:
			min_val = np.mean(val_acc)
			torch.save(model.state_dict(),'best.pkl')
			f = open('record.txt','a')
			f.write("Epoch: {}, Validation : {}\n".format(epoch + 1, min_val))

if __name__ == '__main__':
	main()