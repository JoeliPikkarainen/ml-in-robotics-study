# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:42:38 2023

@author: joeli
"""

from e4 import e4_module_func as mf
from e4 import e4_model_types as mt

def get_avg(rounds,prec,name,model):
    if rounds <= 0:
        return None  # Handle invalid input

    total_test = 0
    total_train = 0
    for _ in range(rounds):
        result_test, result_train = mf(prec,name,model,False)
        total_test += result_test
        total_train += result_train

    average_test = total_test / rounds
    acerage_train = total_test / rounds
    return average_test,acerage_train

lin_reg = mf(0.2,"fruit_data.csv",mt[0],show_plots=True)
svm = mf(0.2,"fruit_data.csv",mt[1],show_plots=True)

avg_lin_test, avg_lin_train = get_avg(100,0.2,"fruit_data.csv",mt[0])
avg_svm_test, avg_svm_trin = get_avg(100,0.2,"fruit_data.csv",mt[1])

print("avg 100 log:(test)"+ str(avg_lin_test))
print("avg 100 log:(train)"+ str(avg_lin_train))

print("avg 100 svm:(test)"+ str(avg_svm_test)) 
print("avg 100 svm:(train)"+ str(avg_svm_trin)) 



