# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:39:31 2023

@author: joeli
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_absolute_error

#%%
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
first_image = x_train[0]
plt.imshow(first_image)
#%%
x_train_flat = x_train.reshape(60000,28*28)/255
x_test_flat = x_test.reshape(10000,28*28)/255
#%%
lb = LabelBinarizer()
ytrainOH = lb.fit_transform(y_train)
ytestOH = lb.fit_transform(y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(28*28))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(x_train_flat,ytrainOH, validation_data=(x_test_flat,ytestOH),
          epochs=10, batch_size=100)
#%%
predictions_test = np.argmax(model.predict(x_test_flat),axis=1)
predictions_train = np.argmax(model.predict(x_train_flat),axis=1)

#%%
wrong_indicies_test = []
for i in range(0,len(y_test)):
    if predictions_test[i] != y_test[i]:
        wrong_indicies_test.append(i)
        
wrong_indicies_train = []
for i in range(0,len(y_test)):
    if predictions_train[i] != y_train[i]:
        wrong_indicies_train.append(i)
        
accu_test = 1.00-(len(wrong_indicies_test) / 10000)
accu_train = 1.00-(len(wrong_indicies_train) / 60000)

print(f"wrong indices test: {len(wrong_indicies_test)} accuracy: {accu_test:.6f}")
print(f"wrong indices train: {len(wrong_indicies_train)} accuracy: {accu_train:.6f}")
#%%
#Print some wrong indicies on test data with plt.imshow(x_test[i])
for i in range(0,len(wrong_indicies_test)):
    if i >= 10:
        break
    idx_of_image = wrong_indicies_test[i]
    wrong_image = x_test[idx_of_image]
    plt.imshow(wrong_image)
    plt.show()

#%%