#necessary imports

import keras
import cv2
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io as spio
from keras.datasets import mnist
import os
from keras.datasets import  mnist
from keras.models import model_from_json

#loading the dataset
(x_train,y_train), (x_test,y_test) =mnist.load_data()
dimData = np.prod(x_train.shape[1:])
x_train = x_train.reshape(x_train.shape[0], dimData)
x_test = x_test.reshape(x_test.shape[0], dimData)

x_train = x_train/255
x_test = x_test/255

# labels should be onehot encoded
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#check the shape of the labels
print (y_train.shape)
print (y_test.shape)

#creating model

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
 
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training model

history_reg = model.fit(x_train, y_train, batch_size=256, epochs=15, verbose=1, 
                            validation_data=(x_test,y_test))
[test_loss, test_acc]=model.evaluate(x_test,y_test)
print('Result is: test_loss :{} and test_acc : {}'.format(test_loss,test_acc))

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history_reg.history['acc'],'r',linewidth=3.0)
plt.plot(history_reg.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)



