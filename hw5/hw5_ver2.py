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
import random


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
	x = np.array(x)
	y = np.array(y)

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

def fgsm_attack(image, alpha, data_grad, epsilon, ori_image):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image - alpha*sign_data_grad

	pert = perturbed_image - ori_image
	pert = torch.clamp(pert,-epsilon,epsilon)

	result = ori_image + pert
	
	# Return the perturbed image
	return result


def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	
	doc = 'labels.csv'
	x_train,y_train = load_data(doc)
	x_ori = x_train.copy()
		
	train_dataset = ImageDataset(x_train,y_train,
									transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
	L_inf = []
	random.seed(87)
	for _, (img, target) in enumerate(train_loader):

		img_cuda = img.float().to(device)
		target_cuda = target.to(device)
		
		output = resnet50(img_cuda)
		predict = torch.max(output, 1)[1]
		acc = np.mean((target_cuda == predict.float()).cpu().numpy())
		train_acc.append(acc)

		img_var = img_cuda.data
		img_var.requires_grad_()
		#print(img_var)

		for i in range(5):
			
			output = resnet50(img_var)

			y_LL = torch.min(output, 1)[1]
			
			loss = loss_fn(output, y_LL)

			resnet50.zero_grad()
			
			loss.backward()
			
			delta_x = img_var.grad.data

			#1,0.095,85%
			#1,0.0987,87.5%
			alpha = 1
			epsilon = 0.08
			result = fgsm_attack(img_var, alpha, delta_x, epsilon, img_cuda)
			img_var.data = result
			
			#print(img_cuda)
		
		
		

		output2 = resnet50(result)

		predict2 = torch.max(output2, 1)[1]
		acc2 = np.mean((target_cuda == predict2.float()).cpu().numpy())
		if acc2 == 1.0:
			guess = random.randint(0,1000)
			#print(guess)
			img_var = img_cuda.data
			img_var.requires_grad_()
			for i in range(5):
			
				output = resnet50(img_var)

				#y_LL = torch.min(output, 1)[1]
				y_LL = torch.tensor([guess]).to(device)
				
				loss = loss_fn(output, y_LL)

				resnet50.zero_grad()
				
				loss.backward()
				
				delta_x = img_var.grad.data

				#1,0.095,85%
				#1,0.0987,87.5%
				alpha = 1
				epsilon = 0.08
				result = fgsm_attack(img_var, alpha, delta_x, epsilon, img_cuda)
				img_var.data = result
		
		output2 = resnet50(result)
		predict2 = torch.max(output2, 1)[1]
		acc2 = np.mean((target_cuda == predict2.float()).cpu().numpy())
		attack_acc.append(acc2)
		#print(predict[0].item(),predict2[0].item())
		
		
		result = result.detach().cpu().squeeze()

		result[0] = result[0]*0.229+0.485
		result[1] = result[1]*0.224+0.456
		result[2] = result[2]*0.225+0.406

		result = result*255.0
		result = torch.clamp(result,0,255)

		result = result.numpy().swapaxes(0, 1).swapaxes(1, 2)
		
		L_inf.append(np.amax(result.astype(int) - x_ori[_].astype(int)))
		
		
		
		img = Image.fromarray(result.astype('uint8')).convert('RGB')
		img.save("{}/{}".format(sys.argv[2],files[_]))
		print('{} is completed!'.format(files[_]))
		
		
		
	print("Acc: {:.4f}".format(np.mean(train_acc)))
	print("Acc: {:.4f}".format(1.0-np.mean(attack_acc)))
	print("L_inf: {:.4f}".format(np.mean(L_inf)))





if __name__ == "__main__": main()