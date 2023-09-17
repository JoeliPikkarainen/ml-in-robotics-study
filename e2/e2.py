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
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.decomposition import PCA
from sklearn.utils import Bunch

import datetime
from mlxtend.plotting import plot_decision_regions

e2_model_types = ["LogisticRegression","SupportVectorModel"]

def e2_module_func(arg_train_prec,arg_file_name,model_type):
    #Parse inputs
    in_train_prec = arg_train_prec
    in_file_name = arg_file_name
    
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
    """
    if model_type == "LogisticRegression":
        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    elif model_type == "SupportVectorModel":
        model = SVC();
    else:
        print("Error: unsupported model_type " + model_type)
        return -1;
    
    print("Using " + model_type)

    model.fit(x_train, y_train)
    df['Predict'] = enc.inverse_transform(model.predict(x_scaled))

    acc_train = accuracy_score(y_train, model.predict(x_train))
    
    time_start = datetime.datetime.now()
    acc_test = accuracy_score(y_test, model.predict(x_test))
    time_end = datetime.datetime.now()
    time_delta = time_end - time_start
    
    print("Accuracy of prediction in training" , acc_train)
    print("Accuracy of prediction in testing" , acc_test)
    
    cm = confusion_matrix(y_test,model.predict(x_test))
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

    