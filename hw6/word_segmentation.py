import jieba
import sys
import csv
import re

def remove_punctuation(line):
	rule = re.compile(u'[^a-zA-Z0-9\u4e00-\u9fa5]')
	line = rule.sub('',line)
	return line


print('Start word_segmentation ...')
jieba.load_userdict(sys.argv[2])

x_id = []
comment = []
with open(sys.argv[1], newline='', encoding='utf-8') as csvfile:
	rows = csv.DictReader(csvfile)
	for row in rows:
		x_id.append(row['id'])
		comment.append(row['comment'])

word_segment = []
for i in range(len(comment)):
	seg_list = jieba.cut(comment[i], cut_all=False)
	word_segment.append(seg_list)

output = open('test_x.txt', 'w', encoding='utf-8')
for i in range(len(word_segment)):
	for word in word_segment[i]:
		if word[0] == 'B' or word[0] == 'b':
			continue
		word = remove_punctuation(word)
		if word != ' ' and word != '':
			output.write(word + ' ')
	output.write('\n')