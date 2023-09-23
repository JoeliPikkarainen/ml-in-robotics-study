# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:39:31 2023

@author: joeli
"""
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD

import cv2 #conda install -c conda-forge opencv

from sklearn.model_selection import train_test_split

#%%
#Set your own path!
data_path = "..\\..\\..\\dogs-vs-cats\\train\\"
filenames = os.listdir(data_path)
labels = [x.split(".")[0] for x in filenames]
data = pd.DataFrame({"filename": filenames, "label": labels})
image_resox = 128
image_resoy = 128

#%%
#Reshape data
target_resolution = (image_resox, image_resoy)
resized_data = []
label_encoder = LabelEncoder()
label_encoder.fit(["cat","dog"])
for index, row in data.iterrows():
    filename = data_path + row["filename"]
    label = row["label"]

    # Read the image using OpenCV
    print("reading image %s" % filename)
    image = cv2.imread(filename)

    if image is not None:
        # Resize the image to the target resolution
        resized_image = cv2.resize(image, target_resolution)
        label_enum = label_encoder.transform([label])
        # Append the resized image and label to the new 2D array
        resized_data.append({"filename": filename, "label": label,"label_enum":label_enum, "resized_image": resized_image})

#%%
#Separate cats and dogs
cat_images = [item for item in resized_data if item['label'] == 'cat']
dog_images = [item for item in resized_data if item['label'] == 'dog']

#Separate them on test data, keep controlled randomness (random_state)
cat_images_train, cat_images_test = train_test_split(cat_images, test_size=2000, random_state=42)
dog_images_train, dog_images_test = train_test_split(dog_images, test_size=2000, random_state=42)

#Put them together in test and train data
train_data = cat_images_train + dog_images_train
test_data = cat_images_test + dog_images_test        
#%%
#Make the x,y arrays
x_train = np.array([item['resized_image'] for item in train_data])
y_train = np.array([item['label_enum'] for item in train_data])
x_test = np.array([item['resized_image'] for item in test_data])
y_test = np.array([item['label_enum'] for item in test_data])

first_image = x_train[0]
plt.imshow(first_image)
plt.show()
#%%
#x_train_flat = x_train.reshape(2000+2000,image_resox,image_resoy,3)/255
#x_test_flat = x_test.reshape(250000 - 4000,image_resox,image_resoy,3)/255

x_train_flat = x_train
x_test_flat = x_test

#%%
lb = LabelBinarizer()
ytrainOH = lb.fit_transform(y_train)
ytestOH = lb.fit_transform(y_test)

model = tf.keras.Sequential()
#Input
model.add(tf.keras.layers.InputLayer((image_resox,image_resoy,3)))

#Conv1
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                 input_shape=(image_resox, image_resoy, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'))

#Conv2
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'))

#Conv2
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

#%%
model.fit(x_train_flat,ytrainOH, validation_data=(x_test_flat,ytestOH),
          epochs=10, batch_size=100)
#%%
predicted_test_raw = model.predict(x_test_flat)
predicted_train_raw = model.predict(x_train_flat)
#%%
predictions_test = np.round(predicted_test_raw)
predictions_train = np.round(predicted_train_raw)

dog_likelihood = predicted_test_raw[:, 0]

#%%
wrong_indicies_test = []
right_indicies_test = []
for i in range(0,len(y_test)):
    if predictions_test[i] != y_test[i]:
        wrong_indicies_test.append(i)
    else:
        right_indicies_test.append(i)
        
wrong_indicies_train = []
for i in range(0,len(y_test)):
    if predictions_train[i] != y_train[i]:
        wrong_indicies_train.append(i)
        
accu_test = 1.00-(len(wrong_indicies_test) / (len(x_test)))
accu_train = 1.00-(len(wrong_indicies_train) / (len(x_train)))

print(f"wrong indices test: {len(wrong_indicies_test)} accuracy: {accu_test:.6f}")
print(f"wrong indices train: {len(wrong_indicies_train)} accuracy: {accu_train:.6f}")
#%%
#Print some wrong indicies on test data with plt.imshow(x_test[i])
for i in range(0,len(wrong_indicies_test)):
    if i >= 6:
        break
    idx_of_image = wrong_indicies_test[i]
    wrong_image = x_test[idx_of_image]
    dog_like = dog_likelihood[idx_of_image]
    cat_like = 1.0 - dog_like
    text = "Dog likely hood %.2f\nCat likely hood %.2f\nWRONG" % (dog_like,cat_like)
    plt.text(10, image_resoy + 40, text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.imshow(wrong_image)
    plt.show()

#%%
for i in range(0,len(right_indicies_test)):
    if i >= 6:
        break
    idx_of_image = right_indicies_test[i]
    wrong_image = x_test[idx_of_image]
    dog_like = dog_likelihood[idx_of_image]
    cat_like = 1.0 - dog_like
    text = "Dog likely hood %.2f\nCat likely hood %.2f\nRIGHT" % (dog_like,cat_like)
    plt.text(10, image_resoy + 40, text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.imshow(wrong_image)
    plt.show()
#%%