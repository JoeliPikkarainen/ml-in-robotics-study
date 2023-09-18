# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:43:31 2023

@author: joeli
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt

import seaborn as sn

mlp_train_log = []

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        #print(f"End of epoch {epoch}. Logs: {logs} acc {val_accuracy}")
        val_accuracy = logs['val_accuracy']
        mlp_train_log.append((epoch,val_accuracy))


e5_model_types = ["LogisticRegression","SupportVectorModel", "MLP"]

def e5_module_func(arg_train_prec, arg_file_name, arg_model_type, show_plots=True):
    in_train_prec = arg_train_prec
    in_file_name = arg_file_name
    model_type = arg_model_type
    #Assignment: test lin-reg and svm
    mlp_train_log.clear()
    print("Reading datafile " + in_file_name)
    df = pd.read_csv(in_file_name)
    
    """
        The data has fields:
        fruit_name(text),
        fruit_subtype(text),
        mass(int),
        width(float),
        height(float),
        color_score(float),

    Use these in classification.
    The data fields need to be "normalized
    
    We must classfy "fruit_name" with
    mass,width,height and color_score.
    
    """
    
    #Read the fields used in classification
    input_vars = ["mass","width","height","color_score"]
    #Used fields to numpy array
    x = np.array(df[input_vars])

    #Reads the "fruit_name" column from csv
    enc = preprocessing.LabelEncoder()
    #Classified fruit_names to ints
    y = enc.fit_transform(np.array(df['fruit_name']))
    
    #Standatdize unit variance
    scaler = preprocessing.StandardScaler()
    #Scale variance
    x_scaled = scaler.fit_transform(x)
    
    #Split data to training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=in_train_prec)
    
    #Create model depending on user input
    if model_type == "LogisticRegression":
        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    elif model_type == "SupportVectorModel":
        model = SVC();
    elif model_type == "MLP":
        model = create_neural_network(df,x,x_scaled,in_train_prec)
    else:
        print("Error: unsupported model_type " + model_type)
        return -1,-1;
    
    print("Using " + model_type)
    
    if model_type != "MLP":
        model.fit(x_train, y_train)
        df['Predict'] = enc.inverse_transform(model.predict(x_scaled))
        
        acc_train = accuracy_score(y_train, model.predict(x_train))
        acc_test = accuracy_score(y_test, model.predict(x_test))
    else:
        y_true_train = y_train#np.argmax(y_train, axis=1)
        y_pred_train = np.argmax(model.predict(x_train), axis=1)
        acc_train = accuracy_score(y_true_train,y_pred_train)
        
        y_true_test = y_test#np.argmax(y_train, axis=1)
        y_pred_test = np.argmax(model.predict(x_test), axis=1)
        acc_test = accuracy_score(y_true_test,y_pred_test)
        
        plot_learning(mlp_train_log)

    print("Accuracy of prediction in training" , acc_train)
    print("Accuracy of prediction in testing" , acc_test)
    
    if show_plots == False:
        return acc_test, acc_train
    
    if model_type != "MLP":
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
    else:
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred_test)
    
    plt.figure()
    ax = plt.axes()
    sn.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='.0f', ax=ax,
               xticklabels=enc.inverse_transform([0,1,2,3]),
               yticklabels=enc.inverse_transform([0,1,2,3])
               )
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.xticks(rotation=0)
    
    info_text = "precent:"+f"{in_train_prec:.2}"+"\nsamples:"+str(len(x_test))+"\nacc_train:"+f"{acc_train:.2f}"+"\nacc_test:"+f"{acc_test:.2f}"
    info_text += "\nmodel:"+model_type
    plt.text(0.5,6, info_text, fontsize=12, ha="center")
    plt.show()
    
    return acc_test, acc_train

def create_neural_network(df,x,x_scaled,in_train_prec):
    y = np.array(pd.get_dummies(df['fruit_name']))
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=in_train_prec)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100,activation='relu', input_shape=(x.shape[1],)),
        tf.keras.layers.Dense(50,activation='relu'),
        tf.keras.layers.Dense(50,activation='relu'),
        tf.keras.layers.Dense(4,activation='softmax'),
                                 ])
    model.compile(loss='categorical_crossentropy',optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
  
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=100, callbacks=MyCallback())
    return model

def plot_learning(epoch_acc):
        x_values, y_values = zip(*epoch_acc)
        pair_with_max_float = max(epoch_acc, key=lambda pair: pair[1])
        y_max = pair_with_max_float[1]
        x_max = pair_with_max_float[0]

        plt.figure(figsize=(8, 6))  # Optional: Set the figure size
        plt.plot(x_values, y_values, marker='o', linestyle='-')  # Customize the marker and linestyle as needed
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        info = "best: " + f"{y_max:.2}"
        info += " at epoch: " + str(x_max)
        plt.title(info)

        plt.grid(True)
        plt.show()