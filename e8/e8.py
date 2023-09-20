# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:56:10 2023

@author: joeli
"""
import pandas as pd
import numpy as np

from sklearn import linear_model

import matplotlib.pyplot as plt

def predict_liner_regression(data,pred_x):
    
    #Make x and y columns
    df = df = pd.DataFrame(data, columns=['x', 'y'])
    
    #Make np array and resize
    x = np.array(df['x'])
    x = x.reshape(-1,1)
 
    #Make np array and resize
    y = np.array(df['y'])
    y = y.reshape(-1,1)
    
    #Format prediction x array
    pred_x = np.array(pred_x)
    pred_x = pred_x.reshape(-1,1)

    #Train with all data
    df_train = df

    model = linear_model.LinearRegression()
    model.fit(x,y)
    
    #Predict with input array
    predict_y = model.predict(pred_x)
    
    #Plot results
    predict_array = [pred_x,predict_y]
    plt.rcParams['figure.dpi'] = 512
    plt.scatter(x, y, color='black', s=3)
    plt.scatter(pred_x, predict_y, color='red', s=3)
    plt.show()
    
    print(predict_y)
    
    return predict_y

