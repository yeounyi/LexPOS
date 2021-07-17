# pip install annoy
# pip install spacy
# python -m spacy download en
# pip install gensim


from annoy import AnnoyIndex
import numpy as np
from collections import Counter
import spacy
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

nlp = spacy.load("en_core_web_lg")

t = AnnoyIndex(50, metric="euclidean")

words = list()
lookup = dict()

for i, line in enumerate(open("/home/yeoun/BART/similar_words/cmudict-0.7b-simvecs", encoding="latin1")):
    word, vals_raw = line.split("  ")
    word = word.lower().strip("(012)")
    vals = np.array([float(x) for x in vals_raw.split(" ")])
    t.add_item(i, vals)
    words.append(word.lower())
    lookup[word.lower()] = i
    
t.build(100)

# load pre-trained word-vectors from gensim-data
word_vectors = api.load("glove-wiki-gigaword-100")

space = dict()
for line in open("/home/yeoun/BART/similar_words/cmudict-0.7b-simvecs", encoding="latin1"):
    line = line.strip()
    word, vec_raw = line.split("  ")
    word = word.lower()
    space[word] = np.array([float(x) for x in vec_raw.split()])


def nnslookup(t, nlp, words, vec, n=100):
    res = t.get_nns_by_vector(vec, n)
    batches = []
    current_batch = []
    last_vec = None
    for item in res:
        if last_vec is None or t.get_item_vector(item) == last_vec:
            current_batch.append(item)
            last_vec = t.get_item_vector(item)
        else:
            batches.append(current_batch[:])
            current_batch = []
            last_vec = None
    if len(current_batch) > 0:
        batches.append(current_batch[:])
    output = []
    for batch in batches:
        output.append(sorted(batch, key=lambda x: nlp.vocab[words[x]].prob, reverse=True)[0])
    return output




def get_phonetically_sim_words_and_scores(keyword, size):
    candidates = [words[i] for i in nnslookup(t, nlp, words, t.get_item_vector(lookup[keyword]), n=size)]

    scores = []
    sim_words = []

    for candidate in candidates:
        try: 
            scores.append(cosine(space[keyword], space[candidate]))
            sim_words.append(candidate)

        except KeyError:
            pass

    df = pd.DataFrame()

    df['words'] = sim_words
    df['phonetic_similarity'] = scores

    return df 


# cosine similarity
def cosine(v1, v2):
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0



# size 클수록 더 많은 단어 보여줌 
def get_phon_sim_words(size, keyword):
    return [words[i] for i in nnslookup(t, nlp, words, t.get_item_vector(lookup[keyword]), n=size)]




def get_sem_sim_words(keywords):
    flatten = lambda l: [item for sublist in l for item in sublist]
    sim_words = []
    for keyword in keywords:
        sim_words.append([word[0] for word in word_vectors.most_similar(keyword)])
    return flatten(sim_words)


# candidates 안의 단어를 keyword와 발음이 유사한 순서대로 정렬 
def sort_by_phon_similarity(candidates, space, keyword):
    
    df = pd.DataFrame()
    
    sim_words = []
    scores = []
    
    try:
        for candidate in candidates:
            scores.append(cosine(space[keyword], space[candidate]))
            sim_words.append(candidate)
    except KeyError:
        pass

    df['words'] = sim_words
    df['phonetic_similarity'] = scores
    
    df = df.sort_values(by="phonetic_similarity", ascending=False)

    return df



# 의미상 관련있는 단어들 중 키워드랑 가장 발음 비슷한 단어 찾기 
def get_keyword(keyword, related_words_list, top=10):
    candidates = get_sem_sim_words(related_words_list)
    return sort_by_phon_similarity(candidates, space, keyword)[:top]



# https://stackoverflow.com/questions/55188209/use-word2vec-to-determine-which-two-words-in-a-group-of-words-is-most-similar

# 발음 유사한 단어들 중 키워드와 가장 의미상 연관되어 있는 단어 찾기 
def get_keyword2(keyword, topn=10):

    candidate_words =  get_phon_sim_words(100, keyword)

    all_pairs = []
    for c in candidate_words:
        all_pairs.append((keyword, c))


    scored_pairs = []
    for p in all_pairs:
        try:
            scored_pairs.append([(word_vectors.similarity(p[0], p[1]), p)])
        except KeyError:
            pass

    sorted_pairs = sorted(scored_pairs, reverse=True)[:topn]

    return sorted_pairs




def get_phonetically_sim_words_and_scores(keyword, size):
    candidates = [words[i] for i in nnslookup(t, nlp, words, t.get_item_vector(lookup[keyword]), n=size)]

    scores = []
    sim_words = []

    for candidate in candidates:
        try: 
            scores.append(cosine(space[keyword], space[candidate]))
            sim_words.append(candidate)

        except KeyError:
            pass

    df = pd.DataFrame()

    df['words'] = sim_words
    df['phonetic_similarity'] = scores

    return df 




def get_similar_words_and_scores(keyword, phon_df):

    df = pd.DataFrame()
    
    candidate_words = phon_df.words.tolist()

    all_pairs = []
    for c in candidate_words:
        all_pairs.append((keyword, c))

    candidates = []
    sem_sim = []
    phon_sim = []
    
    for i,p in enumerate(all_pairs):
        try:
            sem_sim.append(word_vectors.similarity(p[0], p[1]))
            candidates.append(p[1])
            phon_sim.append(phon_df.phonetic_similarity.tolist()[i])
        except KeyError:
            pass

    
    df['words'] = candidates
    df['phonetic_similarity'] = phon_sim
    df['semantic_similarity'] = sem_sim

    
    # df = df.sort_values(["phonetic_similarity", "semantic_similarity"], ascending = (False, False))
    df = df.sort_values(["semantic_similarity", "phonetic_similarity"], ascending = (False, False))
    
    # title = "Semantically and Phonetically Similar Words with " + keyword.upper()
    # df = df.style.set_caption(title)

    return df



# 키워드와 발음 유사한 단어들의 목록과 발음 유사도 계산
# 키워드와 발음 유사한 단어들의 의미 관련도 계산 
# 의미 관련도 & 발음 유사도 2가지 기준으로 sort 
def get_phon_and_sem_similar_words(keyword, n=100):
    phon_df = get_phonetically_sim_words_and_scores(keyword, 100)
    final_df = get_similar_words_and_scores(keyword, phon_df)
    result = []
    for row in final_df.iterrows():
        # BART tokenizer vocab에 있는 단어 & semantic similarity가 양수인 단어 
        if 'Ġ' + row[1][0] in tokenizer.encoder and row[1][2] > 0:
            if keyword in row[1][0]:
                continue
            result.append(row[1][0])
    
    return result 




