# -*- coding: utf-8 -*-
"""
Neural Net 계층 test 직접연습

역전파 고려하지 않는 버전 
"""

import numpy as np 


class Sigmoid:
    def __init__(self):
        self.params = [] 
        
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    
class Affine: 
    def __init__(self, W,b):
        self.params = [W,b]
        
    def forward(self, x):
        W, b = self.params 
        return np.matmul(x,W) + b
    
    
    
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
