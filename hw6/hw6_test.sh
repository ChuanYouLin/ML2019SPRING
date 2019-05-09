wget -O word2vec6.model https://www.dropbox.com/s/93vsznlmw3s2bm2/word2vec6.model?dl=1
wget -O linear_model.npy https://www.dropbox.com/s/ciccw45juq9nfek/linear_model.npy?dl=1
wget -O rnn_02_wv06.pkl https://www.dropbox.com/s/md5q7tovlpsh2d7/rnn_02_wv06.pkl?dl=1
python word_segmentation.py $1 $2
python hw6_test.py $3
