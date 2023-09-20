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

from sklearn import linear_model

def forecast_demand_linreg(in_file_name, in_split,in_future_days):
    #format Date,Open,High,Low,Close,Volume
    #We are forecasting "Close"
    df = pd.read_csv(in_file_name,sep=';')
       
    df['CloseFuture'] = df['Demand'].shift(in_future_days)
    
    #Split to test and train
    df_train = df[:in_split]
    df_test = df[in_split:]
    
    x = np.array(df_train['Day'])
    x = x.reshape(-1,1)
    y = np.array(df_train['Demand'])
    
    model = linear_model.LinearRegression()
    model.fit(x,y)
    
    y = np.array(df_train['CloseFuture'])

    predict_train = model.predict(x)
    df_train['Ennuste'] = predict_train
    
    x_test = np.array(df_test['Day'])
    x_test = x_test.reshape(-1,1)
    predict_test = model.predict(x_test)
    df_test['Ennuste'] = predict_test
    
    plt.scatter(df['Day'].values, df['Demand'].values, color='black',s=1)
    plt.plot(df_train['Day'] + in_future_days , df_train['Ennuste'].values, color='blue')
    plt.plot(df_test['Day']+ in_future_days, df_test['Ennuste'].values, color='red')
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
    
def forecast_demand_mlp(in_file_name, in_split,in_future_days):
    #format Date,Open,High,Low,Close,Volume
    #We are forecasting "Close"
    df = pd.read_csv(in_file_name,sep=';')
       
    df['CloseFuture'] = df['Demand'].shift(in_future_days)
    
    #Split to test and train
    df_train = df[:in_split]
    df_test = df[in_split:]
    
    x = np.array(df_train['Day'])
    x = x.reshape(-1,1)
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    y = np.array(df_train['Demand'])

    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(16, activation='swish'),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.005),
                  loss='mse',#Mean Squared Error
                  metrics=['mae']#Mean Absolute Error
                  )
    
    model.fit(x_scaled,y,epochs=200, batch_size=5)
    
    y = np.array(df_train['CloseFuture'])

    predict_train = model.predict(x_scaled)
    df_train['Ennuste'] = predict_train
    
    x_test = np.array(df_test['Day'])
    x_test = x_test.reshape(-1,1)
    x_test_scaled = scaler.transform(x_test)
    predict_test = model.predict(x_test_scaled)
    df_test['Ennuste'] = predict_test
    
    plt.scatter(df['Day'].values, df['Demand'].values, color='black',s=1)
    plt.plot(df_train['Day'] + in_future_days , df_train['Ennuste'].values, color='blue')
    plt.plot(df_test['Day']+ in_future_days, df_test['Ennuste'].values, color='red')
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


