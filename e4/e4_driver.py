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

    total = 0
    for _ in range(rounds):
        result = mf(prec,name,model,False)
        total += result

    average = total / rounds
    return average

lin_reg = mf(0.2,"fruit_data.csv",mt[0],show_plots=True)
svm = mf(0.2,"fruit_data.csv",mt[1],show_plots=True)

avg_lin = get_avg(100,0.2,"fruit_data.csv",mt[0])
avg_svm = get_avg(100,0.2,"fruit_data.csv",mt[1])

print("avg 100 log:"+ str(avg_lin))
print("avg 100 svm:"+ str(avg_svm)) 



