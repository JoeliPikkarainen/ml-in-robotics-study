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
    df_train['Ennuste_test'] = predict_train
    
    x_test = np.array(df_test[['Time','Close']])
    x_train = np.array(df_train[['Time','Close']])

    predict_test = model.predict(x_test)
    predict_train = model.predict(x_train)

    df_test['Ennuste_test'] = predict_test
    df_train['Ennuste_train'] = predict_train

    plt.scatter(df['Date'].values, df['Close'].values, color='black', s=3)
    plt.plot((df_train['Date'] + pd.DateOffset(days=in_prediction_days)).values , df_train['Ennuste_test'].values, color='blue', lw=1)
    plt.plot((df_test['Date']+ pd.DateOffset(days=in_prediction_days)).values, df_test['Ennuste_test'].values, color='red',lw=1)
    plt.rcParams['figure.dpi'] = 1024

    plt.show()
    
    df_validation_test = df_test.dropna()
    df_validation_train = df_train.dropna()

    mean_abs_error_test = mean_absolute_error(
                        df_validation_test['CloseFuture'],
                        df_validation_test['Ennuste_test'])
    
    mean_abs_error_train = mean_absolute_error(
                        df_validation_train['CloseFuture'],
                        df_validation_train['Ennuste_train'])
    
    print("Prediction mean error is train: %.f test: %.f" %
          (mean_abs_error_test, mean_abs_error_train))
    
    print("Prediction coefficents \n", model.coef_, model.intercept_)
    
    return df_test