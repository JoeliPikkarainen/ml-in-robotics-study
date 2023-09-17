# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:24:07 2023

@author: joeli
"""
import numpy as np
import matplotlib.pyplot as plt

#our own module
from e2 import e2_module_func
from e2 import e2_model_types as mt

print("Executing test module")
#The assignment
assignment_value = e2_module_func(0.2,'sensor_readings_24.csv',mt[1])

#Test other precentages
test_prec_array = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])
                            
ret_vals = []
for value in test_prec_array:
    result = e2_module_func(value,'sensor_readings_24.csv',mt[1])
    ret_vals.append(result)

#Show plot with x=prediction_accuracy, y=precentage of training data in use
plt.plot(test_prec_array, ret_vals, label="Test Prediction accuracy")
plt.xlabel("Precentage of training data in use")
plt.ylabel("Prediction Accuracy")
plt.show()