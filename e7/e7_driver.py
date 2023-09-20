# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:29:51 2023

@author: joeli
"""

from e7 import forecast_stock_price

forecast_7 = forecast_stock_price('Google_Stock_Price.csv', 185, 7)
forecast_60 = forecast_stock_price('Google_Stock_Price.csv', 185, 60)
