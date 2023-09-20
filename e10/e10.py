# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:31:56 2023

@author: joeli
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from sklearn import preprocessing

import matplotlib.pyplot as plt


def forecast_stock_price_mlp(in_file_name, in_split):
    #format Date,Open,High,Low,Close,Volume
    #We are forecasting "Close"
    df = pd.read_csv(in_file_name)
    
    #Date string to number
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df.apply(lambda row: len(df) - row.name, axis=1)
    
    df['CloseFuture'] = df['Close'].shift(30)
    
    #Split to test and train
    df_test = df[:in_split]
    df_train = df[in_split:]
    
    df_train['Delta'] = (df_train['Open'] - df_train['Close'])
    df_train['DeltaFactor'] = df_train['Close'] - (df_train['Delta'])
    
    x = np.array(df_train['Time'])
    x = x.reshape(-1,1)
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    y = np.array(df_train['CloseFuture'])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='sigmoid', input_shape=(1,)),
        tf.keras.layers.Dense(20, activation='sigmoid',),
        tf.keras.layers.Dense(20, activation='relu',),
        tf.keras.layers.Dense(1)
        ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  loss='mse',#Mean Squared Error
                  metrics=['mae']#Mean Absolute Error
                  )
    
    model.fit(x_scaled,y,epochs=100, batch_size=10)
    predict_train = model.predict(x_scaled)
    df_train['Ennuste'] = predict_train
    
    x_test = np.array(df_test['Time'])
    x_test = x_test.reshape(-1,1)
    x_test_scaled = scaler.transform(x_test)
    predict_test = model.predict(x_test_scaled)
    df_test['Ennuste'] = predict_test
    
    plt.scatter(df['Date'].values, df['Close'].values, color='black',s=1)
    plt.plot((df_train['Date'] + pd.DateOffset(days=30)).values , df_train['Ennuste'].values, color='blue')
    plt.plot((df_test['Date']+ pd.DateOffset(days=30)).values, df_test['Ennuste'].values, color='red')
    plt.rcParams['figure.dpi'] = 512
    plt.show()
    
    df_validation = df_test.dropna()
    mean_abs_error_test = mean_absolute_error(
                        df_validation['CloseFuture'],
                        df_validation['Ennuste'])
    
    df_validation = df_train.dropna()
    mean_abs_error_train = mean_absolute_error(
                        df_validation['CloseFuture'],
                        df_validation['Ennuste'])
    
    print("Prediction mean error in test %.f" %
          mean_abs_error_test)
    
    print("Prediction mean error in train %.f" %
          mean_abs_error_train)


