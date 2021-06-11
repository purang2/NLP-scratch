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


