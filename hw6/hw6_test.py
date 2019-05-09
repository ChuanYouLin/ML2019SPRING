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
from hw6_ver2 import Model
import pandas as pd
import os
np.random.seed(1)
torch.cuda.manual_seed_all(1)

def load_data(wv_model,test_x,pad_len):
	wv_dim = 100
	index2word = []
	word2index = {}
	wordvector = []
	model = models.Word2Vec.load(wv_model)
	for i, word in enumerate(model.wv.vocab):
		word2index[word] = len(word2index)
		index2word.append(word)
		wordvector.append(model[word])
	wordvector.append(np.random.uniform(0,1,(wv_dim))) #unk
	wordvector.append(np.random.uniform(0,1,(wv_dim))) #pad
	wordvector = torch.tensor(wordvector, dtype = torch.float)

	x = []
	f = open(test_x,"r")
	for line in f.readlines():
		x.append(line.split())
	

	length = []
	x_vector = []
	

	word_vectors = model.wv

	for i in x:
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

	x_vector_pad = []
	for vector in x_vector:
		max_length = pad_len
		v = np.ones((max_length,))*(len(word2index)+1)
		vector_length = min(vector.shape[0],max_length)
		v[:vector_length] = vector[:vector_length]
		x_vector_pad.append(v)

	return x_vector_pad,length,wordvector

class SentenceDataset(Dataset):
	def __init__(self, x_train, length, transform=None):
		self.x_train = x_train
		self.length = length
	def __len__(self):
		return len(self.x_train)
	def __getitem__(self,idx):
		return self.x_train[idx],self.length[idx]

def main():
	print("Start testing ...")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	device = torch.device('cuda')
	x_test,length,embedding = load_data('word2vec6.model','test_x.txt',32)
	
	
	train_dataset = SentenceDataset(x_test,length)
	
	train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=False)
	
	model = Model(embedding)
	model.to(device)
	model.load_state_dict(torch.load('rnn_02_wv06.pkl'))
	model.eval()
	model2 = np.load("linear_model.npy")


	ans = []
	for _, (sentence,length) in enumerate(train_loader):

		sentence_cuda = sentence.to(device, dtype=torch.long)
		length_cuda = length.to(device)
		
		output = model(sentence_cuda)
			
		for a in output:
			if a >= 0.5:
				ans.append(1)
			else:
				ans.append(0)
	ans = model2

	ans_id = np.arange(20000)
	dataframe = pd.DataFrame({'id':ans_id,'label':ans})
	dataframe.to_csv(sys.argv[1],index=False,sep=',')


if __name__ == '__main__':
	main()