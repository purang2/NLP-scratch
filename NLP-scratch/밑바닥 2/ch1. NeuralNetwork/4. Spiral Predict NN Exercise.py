# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:38:09 2021

@author: PC
"""

import sys 
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

import numpy as np 
#import common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.layers import SoftmaxWithLoss,Affine,Sigmoid
from common.optimizer import SGD 

'''
class Sigmoid:
    def __init__(self):
        self.params = [] 
        self.grads = [], []
        self.out = None 
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 
        return out
    
    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out
    
'''    
'''
class Affine: 
    def __init__(self, W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None 
        
    def forward(self, x):
        W, b = self.params 
        out = np.matmul(x,W) + b
        self.x = x
        return out 
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW 
        self.grads[1][...] = db
        return dx
    

'''    



class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size):
        i,h,o = input_size, hidden_size, output_size
        
        
        W1 = 0.01 * np.random.randn(i,h)
        b1 = np.zeros(h)
        W2 = 0.01 * np.random.randn(h,o)
        b2 = np.zeros(o)
        
        
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)]
        self.loss_layer = SoftmaxWithLoss() #⭐
        
        self.params, self.grads =[],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss 
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
        
        

#하이퍼파라미터 
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0 


#데이터 , 모델 
x, t = spiral.load_data() 

model = TwoLayerNet(input_size =2, hidden_size=hidden_size, output_size =3)

optimizer =SGD(lr=learning_rate)


data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = [] 

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]
    
    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        
        total_loss += loss
        loss_count += 1
        
        
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(' | epoch %d | 반복 %d / %d | 손실 %.2f'%(epoch+1, iters+1,max_iters, avg_loss))
            
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0






