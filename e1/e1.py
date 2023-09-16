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
import matplotlib.pyplot as plt
import seaborn as sn

def e1_module_func(arg_train_prec,arg_file_name):
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
    
    """A dichotomous classification consists of two stages: 
        coding (construction of a code matrix) and 
        decoding
        making a decision on the correspondence of an object to a class by analyzing the code matrix
        
        decoding to 0s and 1s
    """
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(x_train, y_train)
    df['Predict'] = enc.inverse_transform(model.predict(x_scaled))
    
    acc_train = accuracy_score(y_train, model.predict(x_train))
    acc_test = accuracy_score(y_test, model.predict(x_test))
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
    
    plt.text(0.5,6, info_text, fontsize=12, ha="center")
    plt.show()
    return acc_test