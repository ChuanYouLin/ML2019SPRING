import torchvision.models as models
import numpy as np
import sys
import csv
import torch
import torch.nn as nn
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os


def load_data(doc):
	x = []
	y = []
	#x_train = np.zeros([18,1])
	with open(doc, newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			y.append(float(row['TrueLabel']))
	
	mypath = sys.argv[1]
	files = sorted(listdir(mypath))
	for f in files:
		# 產生檔案的絕對路徑
		fullpath = join(mypath, f)
		x.append(np.array(Image.open(fullpath)))

	#normalization to 0~1
	x = np.array(x,dtype=float) / 255.0
	y = np.array(y,dtype=int)

	return x,y

class ImageDataset(Dataset):
	def __init__(self, x_train, y_train, transform=None):
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

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image - epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	
	doc = 'labels.csv'
	x_train,y_train = load_data(doc)
	
	train_dataset = ImageDataset(x_train,y_train,
									transforms.Compose([
										transforms.ToTensor(),
										]))

	train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=False)

	resnet50 = models.resnet50(pretrained=True)
	resnet50.to(device)
	loss_fn = nn.CrossEntropyLoss()
	resnet50.eval()

	mypath = sys.argv[1]
	files = sorted(listdir(mypath))
	
	
	train_acc = []
	attack_acc = []
	for _, (img, target) in enumerate(train_loader):

		img_cuda = img.float().to(device)
		target_cuda = target.to(device)
		img_cuda.requires_grad_()

		output = resnet50(img_cuda)

		predict = torch.max(output, 1)[1]
		acc = np.mean((target_cuda == predict).cpu().numpy())
		train_acc.append(acc)

		loss = loss_fn(output, target_cuda)*(-1.0)

		resnet50.zero_grad()
		
		loss.backward()
		
		delta_x = img_cuda.grad.data

		epsilon = 0.007
		result = fgsm_attack(img_cuda, epsilon, delta_x)

		output2 = resnet50(result)

		predict2 = torch.max(output2, 1)[1]
		acc2 = np.mean((target_cuda == predict2).cpu().numpy())
		attack_acc.append(acc2)
		
		result = result.detach().cpu().numpy()
		result = result.squeeze().swapaxes(0, 1).swapaxes(1, 2)
		for i in range(result.shape[0]):
			for j in range(result.shape[1]):
				for k in range(result.shape[2]):
					result[i,j,k] *= 255.0
		img = Image.fromarray(result.astype('uint8')).convert('RGB')
		img.save("{}/{}".format(sys.argv[2],files[_]))
		print('{} is completed!'.format(files[_]))
		
	print("Acc: {:.4f}".format(np.mean(train_acc)))
	print("Acc: {:.4f}".format(np.mean(attack_acc)))





if __name__ == "__main__": main()