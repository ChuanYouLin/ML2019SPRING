from gensim.models import word2vec
from gensim import models
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
import torch.nn as nn
from torch.optim import Adam,SGD
import sys
import csv
import random
import os
#from matplotlib import pyplot as plt
#random.seed(87)
np.random.seed(1)
torch.cuda.manual_seed_all(1)


def load_data():
	wv_dim = 100
	index2word = []
	word2index = {}
	wordvector = []
	model = models.Word2Vec.load('word2vec6.model')
	for i, word in enumerate(model.wv.vocab):
		word2index[word] = len(word2index)
		index2word.append(word)
		wordvector.append(model[word])
	wordvector.append(np.random.uniform(0,1,(wv_dim))) #unk
	wordvector.append(np.random.uniform(0,1,(wv_dim))) #pad
	wordvector = torch.tensor(wordvector, dtype = torch.float)

	

	x = []
	f = open("train_x.txt","r")
	for line in f.readlines():
		x.append(line.split())
	y = []
	with open(sys.argv[1], newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			y.append(int(row['label']))

	length = []
	x_vector = []
	y_vector = []

	word_vectors = model.wv
	
	fuck = 0
	
	for i in x[:119018]:
		
		vector = []
		for j in i:
			if j in word_vectors.vocab:
				index = word2index[j]
				vector.append(index)
			else:
				vector.append(len(word2index))
		if vector == []:
			vector.append(len(word2index)+1)
		vector = np.array(vector)
		length.append(vector.shape[0])
		x_vector.append(vector)
		y_vector.append(y[fuck])
		fuck += 1

	x_vector_pad = []
	for vector in x_vector:
		max_length = 32
		v = np.ones((max_length,))*(len(word2index)+1)
		vector_length = min(vector.shape[0],max_length)
		v[:vector_length] = vector[:vector_length]
		x_vector_pad.append(v)

	c = list(zip(x_vector_pad,y_vector,length))
	random.shuffle(c)
	x_vector_pad,y_vector,length = zip(*c)

	split_data = int(len(x_vector_pad)*0.1)
	print(split_data)
	#exit(0)
	
	return x_vector_pad[split_data:],y_vector[split_data:],length[split_data:],x_vector_pad[:split_data],y_vector[:split_data],length[:split_data],wordvector


class Model(nn.Module):
	def __init__(self,embedding):
		super(Model, self).__init__()
		self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
		self.embedding.weight = torch.nn.Parameter(embedding)
		self.embedding.weight.requires_grad = True
		self.lstm = nn.GRU(embedding.size(1), embedding.size(1), 2, batch_first=True, bidirectional=False, dropout=0.7)
		self.lstm2 = nn.GRU(8, 8, 2, batch_first=True, bidirectional=False, dropout=0.4)
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Sequential(
			#nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(embedding.size(1),1),
			nn.Sigmoid(),
		)
		self.maxseq = nn.MaxPool1d(32)

	def init_hidden(self, layer_size, batch_size, hidden_dim):
		#return (torch.zeros(layer_size, batch_size, hidden_dim).cuda(),torch.zeros(layer_size, batch_size, hidden_dim).cuda())
		return torch.nn.init.orthogonal_(torch.empty(layer_size, batch_size, hidden_dim).cuda())
	def forward(self, x):
		out = self.embedding(x)
		out = self.dropout(out)
		self.hidden = self.init_hidden(2,out.shape[0],100)
		out, self.hidden = self.lstm(out, self.hidden)

		out = out.permute(0,2,1)
		out = self.maxseq(out)
		out = out.squeeze(2)
		#out = out.permute(1,0,2)
		#out = out[-1]
		out = self.fc(out)
		
		return out

class SentenceDataset(Dataset):
	def __init__(self, x_train, y_train, length, transform=None):
		self.x_train = x_train
		self.y_train = y_train
		self.length = length
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):
		return self.x_train[idx],self.y_train[idx],self.length[idx]


def evaluation(outputs, labels):
	#outputs => probability (float)
	#labels => labels
	outputs[outputs>=0.5] = 1
	outputs[outputs<0.5] = 0
	correct = torch.sum(torch.eq(outputs, labels)).item()
	return correct

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda')
	x_train,y_train,length,x_val,y_val,length_val,embedding = load_data()
	
	
	train_dataset = SentenceDataset(x_train,y_train,length)
	val_dataset = SentenceDataset(x_val,y_val,length_val)

	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
	val_loader = DataLoader(dataset=val_dataset,batch_size=128,shuffle=False)

	model = Model(embedding)
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.BCELoss()
	best_val = 0.0
	train_his = []
	for epoch in range(10):
		train_loss = []
		train_acc = []
		model.train()
		val_acc = []
		for _, (sentence,target,length) in enumerate(train_loader):
			optimizer.zero_grad()
			#model.hidden = model.init_hidden(2,target.shape[0],8)

			target = target.reshape((target.shape[0],1))
			sentence_cuda = sentence.to(device, dtype=torch.long)
			target_cuda = target.to(device, dtype=torch.float)

			output = model(sentence_cuda)

			lamda = 0.0001
			l2_reg = torch.tensor(0.).to(device)
			for param in model.parameters():
				l2_reg += torch.norm(param)**2

			loss = loss_fn(output, target_cuda) + l2_reg*lamda
			loss.backward()
			optimizer.step()

			
			#predict = torch.max(output, 1)[1]
			#acc = np.mean((target_cuda == predict).cpu().numpy())
			correct = evaluation(output, target_cuda)
			acc = correct/128

			train_acc.append(acc)
			train_loss.append(loss.item())
		model.eval()
		for _, (sentence,target,length) in enumerate(val_loader):
			#model.hidden = model.init_hidden(2,target.shape[0],8)
			target = target.reshape((target.shape[0],1))
			sentence_cuda = sentence.to(device, dtype=torch.long)
			target_cuda = target.to(device, dtype=torch.float)
			output = model(sentence_cuda)
			#predict = torch.max(output, 1)[1]
			#acc = np.mean((target_cuda == predict).cpu().numpy())
			correct = evaluation(output, target_cuda)
			acc = correct/128
			val_acc.append(acc)

		train_his.append(np.mean(train_acc))
		print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Val: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc), np.mean(val_acc)))
		cur_val = np.mean(val_acc)
		if cur_val > best_val:
			best_val = cur_val
			torch.save(model.state_dict(),'rnn_02_wv06.pkl')
			print('Model Saved!')
	#arrX = np.arange(0,10)
	#plt.plot(arrX,train_his)
	#plt.savefig("test.jpg")


if __name__ == '__main__':
	main()
