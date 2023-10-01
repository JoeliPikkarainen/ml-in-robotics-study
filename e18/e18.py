# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 14:45:12 2023

@author: joeli
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

plt.ion()

in_file_name = "process_data.csv"
in_split = 0

prediction_parameters = {"T_data_1_1","T_data_1_2","T_data_1_3","T_data_2_1","T_data_2_2","T_data_2_3","T_data_3_1","T_data_3_2","T_data_3_3","T_data_4_1","T_data_4_2","T_data_4_3","T_data_5_1","T_data_5_2","T_data_5_3","H_data","AH_data"}
prediction_parameters_out = {"quality"}

prediction_parameters2 = {"T_data_1_1","T_data_1_2","T_data_1_3","T_data_2_1","T_data_2_2","T_data_2_3","T_data_3_1","T_data_3_2","T_data_3_3","T_data_4_1","T_data_4_2","T_data_4_3","T_data_5_1","T_data_5_2","T_data_5_3","AH_data"}

#H_data = thickness reverse predict this
#AH_data = dampness include this

df = pd.read_csv(in_file_name,sep=',')

X = np.array(df.drop(['date_time','quality'], axis = 1))
y = np.array(df['quality'])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

predict_train = model.predict(x_train)
predict_test = model.predict(x_test)

mean_abs_error_train = mean_absolute_error(
                    predict_train,
                    y_train)

mean_abs_error_test = mean_absolute_error(
                    predict_test,
                    y_test)

print("Prediction mean error in train %.f" %
      mean_abs_error_train)

print("Prediction mean error in test %.f" %
      mean_abs_error_test)



#PLOT

plt.figure(1)
plt.title("TRAIN")
x = np.arange(len(predict_train))
plt.plot(x, predict_train, label='PREDICT',color='red', linestyle='-',linewidth=1)
plt.plot(x, y_train, label='ACTUAL', color='blue', linestyle='--',linewidth=1)
plt.rcParams['figure.dpi'] = 1024
plt.legend()
plt.show()


"""
plt.figure(2)
plt.title("TEST")
x = np.arange(len(predict_test))
plt.plot(x, predict_test, label='PREDICT',color='red', linestyle='-',linewidth=1)
plt.plot(x, y_test, label='ACTUAL', color='blue', linestyle='--',linewidth=1)
plt.rcParams['figure.dpi'] = 1024
plt.legend()
plt.show()
"""

#Predict the question:
#How much thickness is needed to have quality > 360 with ceratain temperatures
guess_thickness = 0
got_quality = 0.0
while(got_quality < 360):
    guess_thickness += 0.5
    measurements = [231,232,229,301,353,313,530,530,549,350,349,333,212,219,206,guess_thickness,7.31]
    got_quality = model.predict(np.array(measurements).reshape(1,-1))
    
print(f"Thickness needed {guess_thickness} for quality {got_quality}")







