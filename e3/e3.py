# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 17:08:47 2023

@author: joeli
"""
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.utils import Bunch

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


import matplotlib.pyplot as plt
import seaborn as sn

import datetime

e3_model_types = ["LogisticRegression","SupportVectorModel","MLP"]
mlp_train_log = []

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        #print(f"End of epoch {epoch}. Logs: {logs} acc {val_accuracy}")
        val_accuracy = logs['val_accuracy']
        mlp_train_log.append((epoch,val_accuracy))


def e3_module_func(arg_train_prec,arg_file_name,model_type):
    #Parse inputs
    in_train_prec = arg_train_prec
    in_file_name = arg_file_name
    mlp_train_log.clear()
    
    print("Reading sensor " + in_file_name)
    df = pd.read_csv(in_file_name)
    
    input_vars = []
    
    for  i in range(1,df.shape[1]):
        input_vars.append('Sensor'+str(i))
        
    enc = preprocessing.LabelEncoder()
    
    x = np.array(df[input_vars])
    
    #Reads the "Command" column from csv
    y = enc.fit_transform(np.array(df['Command']))
    
    """
    Standardize features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    
    where u is the mean of the training samples or zero if with_mean=False, 
    and s is the standard deviation of the training samples or one if with_std=False.
    """
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    #Split arrays or matrices into random train and test subsets.
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=in_train_prec)
    
    """
        if model_type == "LogisticRegression":

        A dichotomous classification consists of two stages: 
        coding (construction of a code matrix) and 
        decoding
        making a decision on the correspondence of an object to a class by analyzing the code matrix
        
        decoding to 0s and 1s? Between 0s and 1s?
        
        elif model_type == "SupportVectorModel":
        A support vector machine (SVM)
         is a supervised machine learning model that uses classification algorithms for two-group classification problems.
         After giving an SVM model sets of labeled training data for each category, they’re able to categorize new text (properties).
         
         elif model_type == "MLP":
             A neural network

    """
    time_start = datetime.datetime.now()

    if model_type == "LogisticRegression":
        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    elif model_type == "SupportVectorModel":
        model = SVC();
    elif model_type == "MLP":
        y = np.array(pd.get_dummies(df['Command']))
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
    else:
        print("Error: unsupported model_type " + model_type)
        return -1;
    
    print("Using " + model_type)


    if model_type != "MLP":
        model.fit(x_train, y_train)
        df['Predict'] = enc.inverse_transform(model.predict(x_scaled))
        
        acc_train = accuracy_score(y_train, model.predict(x_train))
        acc_test = accuracy_score(y_test, model.predict(x_test))
    else:
        acc_train = accuracy_score(np.argmax(y_train, axis=1),
                                   np.argmax(model.predict(x_train), axis=1))
        acc_test = accuracy_score(np.argmax(y_test, axis=1),
                                   np.argmax(model.predict(x_test), axis=1))
        plot_learning(mlp_train_log)


      
    time_end = datetime.datetime.now()
    time_delta = time_end - time_start
    
    print("Accuracy of prediction in training" , acc_train)
    print("Accuracy of prediction in testing" , acc_test)
    
    if model_type != "MLP":
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
    else:
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        y_test = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_test,y_pred)

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
    info_text += "\ntime(ms):"+ str(time_delta.total_seconds() * 1000)
    plt.text(0.5,6, info_text, fontsize=12, ha="center")
    plt.show()
    
    """
    This plot is not working correctly and is very slow...
    iris = Bunch(a=x_test,b=y_test)
    iris.data = x_train;
    iris.target = y_train;
    e2_show_decision(iris.data[:, :2], iris.target)
    """
    return acc_test

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
    
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def e2_show_decision(X,y):
    model = SVC(kernel='rbf')
    clf = model.fit(X, y)
    
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC ')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y label here')
    ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

    