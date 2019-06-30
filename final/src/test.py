import pandas as pd
import json
import numpy as np 
import sys
from time import time
import jieba
import csv
from gensim.models.word2vec import Word2Vec
import sys

def tf_idf(content_dic, News_URL):
	idf = {}
	cnt = 0.0
	for url in News_URL:
		content = content_dic[url]
		cnt += 1
		for i, word in enumerate(content):
			if word not in content[:i]:
				try:
					idf[word] += 1.0
				except:
					idf[word] = 1.0
	dict_keys = list(idf.keys())
	for key in dict_keys:
		idf[key] = np.log2(cnt/idf[key])
	
	tf = {}
	for i, url in enumerate(News_URL):
		content = content_dic[url]
		length = len(content)
		tmp_dic = {}
		for word in content:
			try:
				tmp_dic[word] += 1.0/length
			except:
				tmp_dic[word] = 1.0/length
		tf[url] = tmp_dic
	return tf, idf

def find_best_data(tf, idf, News_Index, News_URL, q, model):
	q = list(jieba.cut(q))
	q1 = []
	for word in q:
		if len(q1) < 3:
			q1.append(word)
		else:
			tmp_word = word
			for i in range(3):
				if idf[tmp_word] > idf[q1[i]]:
					tmp = q1[i]
					q1[i] = tmp_word
					tmp_word = tmp
	q2 = [] 
	weights = {}
	for word in q1:
		q2.append(word)
		weights[word] = 1.0
		try:
			ms = model.most_similar([word])
			for j in range(4):
				q2.append(ms[j][0])
				weights[ms[j][0]] = ms[j][1]
		except:
			a = 1
	scores = []
	for url in News_URL:
		tmp_score = 0.0
		for word in q2:
			try:
				tmp_score += tf[url][word]*idf[word]*weights[word]
			except:
				a = 1
		scores.append(tmp_score)
	
	rank = np.argsort(scores)[::-1]
	out_index = []
	for i in rank:
		out_index.append(News_Index[i])
	return out_index


if __name__ == '__main__':
	start_time = time()
	content_dic = json.loads(open(sys.argv[1], 'r').read())
	print('Load json time:', time() - start_time)
	start_time = time()

	NC_1 = pd.read_csv(sys.argv[2])
	News_URL = NC_1.News_URL
	News_Index =  NC_1.News_Index
	print('Load NC_1 time:', time() - start_time)
	start_time = time()

	QS_1 = pd.read_csv(sys.argv[3])
	Query_Index = QS_1.Query_Index
	Query = QS_1.Query
	print('Load QS_1 time:', time() - start_time)
	start_time = time()

	jieba.load_userdict(sys.argv[4])

	tf, idf = tf_idf(content_dic, News_URL)
	print('if-idf time:', time() - start_time)
	start_time = time()

	TD = pd.read_csv(sys.argv[5])
	Query_TD = TD.Query
	News_Index_TD = TD.News_Index
	Relevance = TD.Relevance
	print('Load TD time:', time() - start_time)
	start_time = time()

	model = Word2Vec(list(content_dic.values()), size=250, iter=10, sg=1, workers=15)
	model.save('word2vec.model')

	with open(sys.argv[6], 'w') as fp:
		writer = csv.writer(fp)
		col_name = ['Query_Index']
		for i in range(4):
			for j in range(10):
				for k in range(10):
					if i == 0 and j == 0 and k == 0:
						continue
					col_name.append('Rank_' + str(i) + str(j) + str(k))
					if i == 3:
						break
				if i == 3:
					break
		writer.writerow(col_name)

		cnt = 1
		for q in Query:
			col_data = []
			bad_data = []
			if cnt < 10:
				col_data.append('q_0' + str(cnt))
			else:
				col_data.append('q_' + str(cnt))
			
			cnt2 = 0
			for q2, NI_TD, r in zip(Query_TD, News_Index_TD, Relevance):
				if q2 == q and r != '0':
					col_data.append(NI_TD)
					cnt2 += 1
				elif q2 == q:
					bad_data.append(NI_TD)

			best_data = find_best_data(tf, idf, News_Index, News_URL, q, model)
			for index in best_data:
				if index not in col_data and index not in bad_data:
					col_data.append(index)
					cnt2 += 1
				if cnt2 == 300:
					break
			writer.writerow(col_data)
			cnt += 1
