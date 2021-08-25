import os
import pandas as pd
import ast
import spacy
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

nlp = spacy.load("en_core_web_lg")

# csv로 저장해서 string이 된 list 되돌리기
def to_list(x):
    return ast.literal_eval(x)

def pos(slogan):
    pos = []
    doc = nlp(slogan)
    for token in doc:
        if str(token) in ['<', '>']:
            continue
        if str(token) in ['name', 'loc']:
            pos.append("PROPN")
        elif str(token) == 'year':
            pos.append("NUM")
        else:
            tag = token.pos_
            if str(tag) == 'SPACE':
                continue
            pos.append(tag)
    return pos

filenames = ['/home/yeoun/BART/PREDS/SynSemBart_lowered_preds.csv', '/home/yeoun/BART/PREDS/originalBart_preds.csv']

# df = pd.DataFrame()

for i, filename in enumerate(filenames):
	if i == 0:
		df = pd.read_csv(filename)
		df = df[['slogan', 'given_keywords', 'predicted_slogan']]
		df = df.rename(columns={'predicted_slogan':filename})
	else:
		temp = pd.read_csv(filename)
		temp = temp.rename(columns={'predicted_slogan':filename})
		df = pd.concat([df, temp[filename]], axis=1)

df.to_csv('multi_merged_preds.csv', index=False, encoding='utf-8')

N = df.shape[0]
df.given_keywords = df.given_keywords.apply(lambda x: to_list(x))
keywords = df.given_keywords.tolist()
slogans = df.slogan.tolist()

pos_tags = [["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]]
multi = MultiLabelBinarizer()
multi.fit(pos_tags)

for column in filenames:
	preds = df[column].tolist()
	keyword_rates = []
	pos_f1 = []

	# keyword order score 추가
	# keyword
	for keyword, pred, slogan in zip(keywords, preds, slogans):
		doc = nlp(pred.lower())
		tokens = []
		for token in doc:
			tokens.append(str(token))

		total = len(keyword)
		count = 0
		for k in keyword:
			if k in tokens:
				count += 1

		if count / total > 1:
			print(keyword, slogan)

		keyword_rates.append(count / total)

		slogan_pos = [pos(slogan)]
		pred_pos = [pos(pred)]

		A_new = multi.transform(slogan_pos)
		B_new = multi.transform(pred_pos)
		pos_f1.append(f1_score(A_new,B_new,average='samples'))

	print(column, ' 키워드 포함 비율: ', sum(keyword_rates) / N)
	print(column, ' POS f1: ', sum(pos_f1) / N)