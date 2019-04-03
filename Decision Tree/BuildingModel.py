#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:56:04 2018

@author: akankshaupadhyay
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
home_data_file_path = '/Users/akankshaupadhyay/Documents/Kaggle/Assignment_Exploringdataset/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(home_data_file_path)

# Call line below with no argument to check that you've loaded the data correctly
home_data.describe()
#print(home_data.columns)

# What is the average lot size (rounded to nearest integer)?
print("Average lot size", home_data.LotArea.mean())
avg_lot_size = home_data.mean()

# As of today, how old is the newest home (current year - the date in which it was built)
date_built = home_data.YearBuilt
newest_home_age = 2019 - max(home_data.YearBuilt)
print('newest home age = ', newest_home_age,',Date in which it was built = ', max(home_data.YearBuilt))

y = home_data.SalePrice

x_featureData = ['LotArea', 'YearBuilt','1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[x_featureData]
print(X.head())

home_data_model = DecisionTreeRegressor(random_state = 1)
home_data_model.fit(X, y)
print('the predictions are ', home_data_model.predict(X.head()))