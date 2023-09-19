# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:26:54 2023

@author: joeli
"""
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

def forecast_stock_price(in_file_name, in_split,in_prediction_days):
    #format Date,Open,High,Low,Close,Volume
    #We are forecasting "Close"
    df = pd.read_csv(in_file_name)
    
    #Date string to number
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df.apply(lambda row: len(df) - row.name, axis=1)
    df['CloseFuture'] = df['Close'].shift(in_prediction_days)
    
    #Split to test and train
    df_test = df[:in_split]
    df_train = df[in_split:]
    
    x = np.array(df_train[['Time','Close']])
    y = np.array(df_train['CloseFuture'])
    
    model = linear_model.LinearRegression()
    
    model.fit(x,y)
    predict_train = model.predict(x)
    df_train['Ennuste'] = predict_train
    
    x_test = np.array(df_test[['Time','Close']])
    predict_test = model.predict(x_test)
    df_test['Ennuste'] = predict_test
    
    
    #plt.scatter(df['Date'].values, df['Close'].values, color='black', s=3)
    plt.plot((df_train['Date'] + pd.DateOffset(days=in_prediction_days)).values , df_train['Ennuste'].values, color='blue', lw=1)
    plt.plot((df_test['Date']+ pd.DateOffset(days=in_prediction_days)).values, df_test['Ennuste'].values, color='red',lw=1)
    plt.rcParams['figure.dpi'] = 1024

    plt.show()
    
    df_validation = df_test.dropna()
    mean_abs_error = mean_absolute_error(
                        df_validation['CloseFuture'],
                        df_validation['Ennuste'])
    
    print("Prediction mean error is %.f" %
          mean_abs_error)
    
    print("Prediction coefficents \n", model.coef_, model.intercept_)
    
    return 0