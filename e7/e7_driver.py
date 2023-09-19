# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:29:51 2023

@author: joeli
"""

from e7 import forecast_stock_price

forecast_0 = forecast_stock_price('Google_Stock_Price.csv', 185, 0)
forecast_15 = forecast_stock_price('Google_Stock_Price.csv', 185, 15)
forecast_60 = forecast_stock_price('Google_Stock_Price.csv', 185, 60)
forecast_100 = forecast_stock_price('Google_Stock_Price.csv', 185, 100)