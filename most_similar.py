# -*- coding: utf-8 -*-


import sys
sys.path.append('..')
import numpy as np
from util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
C= create_co_matrix(corpus, vocab_size)



print("코퍼스에 대한 각 검색어와 가장 유사한 단어 4개 추출")

for txt in word_to_id:
    most_similar(txt, word_to_id, id_to_word ,C, top=4)
    #most_similar('you', word_to_id, id_to_word ,C, top=5)

