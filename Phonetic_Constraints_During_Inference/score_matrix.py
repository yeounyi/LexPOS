import numpy as np
from numpy.linalg import norm
from numpy import dot
from transformers import BartTokenizer

space = dict()
for line in open("../similar_words/cmudict-0.7b-simvecs", encoding="latin1"):
    line = line.strip()
    word, vec_raw = line.split("  ")
    word = word.lower()
    space[word] = np.array([float(x) for x in vec_raw.split()])


# cosine similarity
def cosine(v1, v2):
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

in_space = []
for word in list(tokenizer.encoder.keys()):
    if word in space or word.replace('Ġ', '') in space:
        in_space.append(word)

# 50265 * 50265
score_matrix = np.zeros((tokenizer.vocab_size,tokenizer.vocab_size))

for key in in_space:
    for candidate in in_space:
        similarity_score = cosine(space[key.replace('Ġ', '')], space[candidate.replace('Ġ', '')])
        score_matrix[tokenizer.encoder[key]][tokenizer.encoder[candidate]] = similarity_score

np.savetxt('score_matrix.txt', np.array(score_matrix))



