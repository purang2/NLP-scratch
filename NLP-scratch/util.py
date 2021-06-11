# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:08:26 2021

@author: PC
"""
import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


#To create Co-occurrence matrix(동시발생행렬)
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size) , dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size +1):
            left_idx = idx -i
            right_idx = idx+i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] +=1
                
    return co_matrix


#Cosine Similarity, 단어 벡터의 유사도 

def cos_similarity(x,y, eps =1e-8):
    
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    
    return np.dot(nx, ny)



def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    
    if query not in word_to_id:
        print("%s(을)를 찾을 수 없습니다." %query)
        return
    
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
    count =0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return 
        