import numpy as np
import sys
import csv
import pandas as pd


def linear_re(doc,ans_file,model):
	x = []
	ans_id = []
	#x_train = np.zeros([18,1])
	with open(doc, newline='',encoding="big5") as csvfile:
		rows = csv.reader(csvfile)
		hour = []
		i = 0
		for row in rows:
			if row[0] not in ans_id:
				ans_id.append(row[0])
			for j in range(len(row)):
				if row[j] == 'NR':
					row[j]=0
			if int(i / 18) == 0:
				x.append(row[2:])
			else:
				for k in row[2:]:
					x[i%18].append(k)
			#print(row)
			i += 1

	x_test_pre = np.array(x,dtype='float64')
	#print(x_test_pre[0])
	
	x_test = []
	
	for data_num in range(int(len(x_test_pre[0])/9)):
		c = x_test_pre[2:,9*data_num:9*(data_num+1)].copy()
		
		
		c_re = np.reshape(c,(c.shape[0]*c.shape[1]))
		c_re = list(c_re)
		c_re.insert(0,1)
		c_re = np.array(c_re)

		x_test.append(c_re)
		

	x_test = np.array(x_test)
	
	#print(len(x_test_pre[0]))
	#print(x_test[1][10:20])
	#print(w.shape)
	w = np.load(model)
	

	ans = x_test.dot(w)
	ans = np.reshape(ans,(1,ans.shape[0]))
	#print(ans)

	for neg in range(len(ans[0])):
		ans[0][neg] = max(ans[0][neg],0)
	

	dataframe = pd.DataFrame({'id':ans_id,'value':list(ans[0])})
	dataframe.to_csv(ans_file,index=False,sep=',')




def main():
	doc = sys.argv[1]
	ans_file = sys.argv[2]
	model = sys.argv[3]
	linear_re(doc,ans_file,model)

if __name__ == "__main__": main()