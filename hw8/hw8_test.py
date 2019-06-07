import numpy as np
import sys
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from hw8_ver2 import MobileNet
import pandas as pd

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

	return x

class ImageDataset(Dataset):
	def __init__(self, x_train, transform=None):
		self.x_train = x_train
		self.transform = transform
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):

		if self.transform:
			sample_x = self.transform(self.x_train[idx])

		return sample_x

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		map_location = "cuda"
	else:
		map_location = "cpu"
	doc = sys.argv[1]
	x_train = load_data(doc)
	
	train_dataset = ImageDataset(x_train,
									transform=transforms.Compose([
										transforms.ToTensor(),
										]))
	
	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=False)

	model = MobileNet()
	model.to(device)
	model.load_state_dict(torch.load('best.pkl', map_location = map_location))
	model.eval()

	result = []
	for _, (img) in enumerate(train_loader):
			img_cuda = img.float().to(device)

			output = model(img_cuda)

			predict = torch.max(output, 1)[1].cpu().numpy()
			for ans in predict:
				result.append(ans)

	ans_id = np.arange(7178)
	dataframe = pd.DataFrame({'id':ans_id,'label':result})
	dataframe.to_csv(sys.argv[2],index=False,sep=',')

if __name__ == '__main__':
	main()