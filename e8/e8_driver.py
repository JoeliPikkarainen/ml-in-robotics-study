# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:56:27 2023

@author: joeli
"""
from e8 import predict_liner_regression

data1 = [
    (1.00, 1.00),
    (2.00, 2.00),
    (3.00, 1.3),
    (4.00, 3.75),
    (5.00, 2.25)
]

pred1 = predict_liner_regression(data1, [6.00])

data2 = [
    (1.0, 1.0),
    (2.0, 2.0),
    (3.0, 3.0),
    (4.0, 4.0)    
    
]

pred1 = predict_liner_regression(data2, [5.0,6.0,7.0,8.0])


print("done")