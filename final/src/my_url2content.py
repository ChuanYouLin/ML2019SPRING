import pandas as pd
import json
import numpy as np 
import sys
from time import time
import jieba

def my_url2content(content_dic, News_URL):
	dict_pos = {}
	word_num = 0
	content_lens = np.zeros((len(News_URL), ))
	for i, url in enumerate(News_URL):
		seg_list = list(jieba.cut(content_dic[url]))
		content_dic[url] = seg_list

	json.dump(content_dic, open(sys.argv[2], 'w'))
	
	return content_dic


if __name__ == '__main__':
	start_time = time()
	content_dic = json.loads(open(sys.argv[1], 'r').read())
	print('Load json time:', time() - start_time)
	start_time = time()

	json.dump(content_dic, open(sys.argv[2], 'w'))
	print('dump time:', time() - start_time)
	start_time = time()

	NC_1 = pd.read_csv(sys.argv[3])
	News_URL = NC_1.News_URL
	News_Index =  NC_1.News_Index
	print('Load NC_1 time:', time() - start_time)
	start_time = time()

	QS_1 = pd.read_csv(sys.argv[4])
	Query_Index = QS_1.Query_Index
	Query = QS_1.Query
	print('Load QS_1 time:', time() - start_time)
	start_time = time()

	jieba.load_userdict(sys.argv[5])

	content_dic_new = my_url2content(content_dic, News_URL)
	print('my_url2content time:', time() - start_time)
	start_time = time()