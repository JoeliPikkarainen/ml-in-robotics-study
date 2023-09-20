# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:42:46 2023

@author: joeli
"""

from e11 import forecast_demand_mlp
from e11 import forecast_demand_linreg

pred_mlp = forecast_demand_mlp('Kysynta.csv',250,50)
pred_linreg = forecast_demand_linreg('Kysynta.csv',250,50)