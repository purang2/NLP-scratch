

# -*- coding: utf-8 -*-
"""
Neural Net 계층 test 직접연습

역전파 고려하는 버전 
"""

import numpy as np 


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
    
    

class Affine: 
    def __init__(self, W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
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
    





# 학습(Train)의 과정: 
#기울기를 구한 뒤, -> 가중치를 갱신해준다 : SGD, [Momentum, AdaGrad, Adam 등]
#오차역전법(Error Backpropagation)에서는 가장 큰 손실을 가리키는 기울기를 얻는다 
#기울기와 반대방향으로 갱신시 오차를 줄일 수 있다.


class SGD: 
    def __init__(self, lr=0.01):
        self.lr = lr 
        
    
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
    


'''


x = np.random.randn(10,2)


w1 = np.random.randn(2,4)
b1 = np.random.randn(4)


w2 = np.random.randn(4,3)
b2 = np.random.randn(3)


L1 = Affine(w1,b1)
L2 = Sigmoid()
L3 = Affine(w2,b2)


y = L1.forward(x)
z = L2.forward(y)
o = L3.forward(z)



'''

# 좀더 간결하게 위의 과정을 만들기 -> 사용할 계층을 모두 단일클래스화 

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        i, h, o = input_size, hidden_size, output_size
        
        W1 = np.random.randn(i,h)
        b1 = np.random.randn(h)
        W2 = np.random.randn(h,o)
        b2 = np.random.randn(o)
        

        self.layers = [Affine(W1,b1), Sigmoid(), Affine(W2,b2)]
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params   # a = [1,2] a+=[3,4] = [1,2,3,4]
            
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x 
    

        
x = np.random.randn(10,2)
net = TwoLayerNet(2,4,3)

o = net.predict(x)
