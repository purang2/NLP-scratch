# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:11:14 2021

@author: PC
"""

from keras.datasets import imdb 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense

training_set, testing_set = imdb.load_data(num_words = 10000)
X_train, y_train  = training_set
X_test, y_test = testing_set


#Zero Padding (모든 리뷰 벡터의 크기를 통일)
X_train_padded = sequence.pad_sequences(X_train, maxlen=100)
X_test_padded = sequence.pad_sequences(X_test, maxlen=100)


def train_model(Optimizer, X_train, y_train, X_val, y_val):
    #Keras Seq 클래스로 모델 선언시 레이어를 손쉽게 쌓을 수 있음
    model = Sequential()
    model.add(Embedding(input_dim= 10000, output_dim = 128))
    model.add(LSTM(units=128))
    #dense -sigmoid의 역할 모델 출력을 0과 1 사이의 확률로 변환시켜주는 것
    model.add(Dense(units=1, activation = 'sigmoid'))
    
    #모델 컴파일 & 훈련
    model.compile(loss='binary_crossentropy',optimizer=Optimizer, metrics=['accuracy'])
    
    #훈련 정확도, 검증 정확도, 에폭별 로스 가ㅓㅄ 등 
    scores = model.fit(X_train,y_train,
                       batch_size = 128, epochs =10,
                       validation_data =(X_val, y_val))
    return scores , model


#LSTM 옵티마이저 성능(궁합) 비교 #1 sgd , #2 RMSprop #3 Adam
sgd_score, sgd_model = train_model(Optimizer='sgd',
                                   X_train=X_train_padded,
                                   y_train=y_train,
                                   X_val=X_test_padded,
                                   y_val=y_test)


rms_score, rms_model = train_model(Optimizer='RMSprop',
                                   X_train=X_train_padded,
                                   y_train=y_train,
                                   X_val=X_test_padded,
                                   y_val=y_test)


adam_score, adam_model = train_model(Optimizer='adam',
                                   X_train=X_train_padded,
                                   y_train=y_train,
                                   X_val=X_test_padded,
                                   y_val=y_test)


from matplotlib import pyplot as plt 



#Plot for SGD optimizer 
plt.plot(range(1,11), sgd_score.history['accuracy'] , label='Traning Accuracy')
plt.plot(range(1,11), sgd_score.history['val_accuracy'] , label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using SGD optimizer')
plt.legend()
plt.show()


#Plot for RMS-prop optimizer 
plt.plot(range(1,11), rms_score.history['accuracy'] , label='Traning Accuracy')
plt.plot(range(1,11), rms_score.history['val_accuracy'] , label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using RMSProp optimizer')
plt.legend()
plt.show()

#Plot for ADAM optimizer 
plt.plot(range(1,11), adam_score.history['accuracy'] , label='Traning Accuracy')
plt.plot(range(1,11), adam_score.history['val_accuracy'] , label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using Adam optimizer')
plt.legend()
plt.show()




#Seaborn 혼동 행렬(Conf matrix) (= 모델의 성능을 한눈에 확인 가능, TN, FP, FN, TP)
#Seaborn = 그림그리는 라이브러리
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(10,7))
sns.set(font_scale=2)
y_test_pred = rms_model.predict_classes(X_test_padded)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Negative Sentiment','Positive Sentiment'],
                 yticklabels=['Negative Sentiment','Positive Sentiment'], cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()