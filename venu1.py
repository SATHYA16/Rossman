# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:30:15 2015

@author: apple
"""
#import re
#import datetime
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Assign the absolute path of the data file to a variable
file_path_train = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656//Project/Rossman/train.csv')
file_path_store = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656//Project/Rossman/store.csv')

#Assign the contents of the datafile to a Pandas dataframe
df_train = pd.read_csv(file_path_train, sep=',', header=0) #Indicate to treat first row as column names
df_store = pd.read_csv(file_path_store, sep=',', header=0) #Indicate to treat first row as column names

print ('START -- Process to convert non-integer labels to integers')

#Categorical variables with non-integer labels mapped to integers
#StateHoliday (in train data set)
#df_train.StateHoliday.unique() = array(['0', 'a', 'b', 'c', 0], dtype=object)
df_train['StateHoliday'] = df_train['StateHoliday'].apply(lambda value: 0 if (value == '0' or value == 0) else 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 999)

#Categorical variables with non-integer labels mapped to integers
#StoreType (in store data set)
#df_store.StoreType.unique() = array(['c', 'a', 'd', 'b'], dtype=object)
df_store['StoreType'] = df_store['StoreType'].apply(lambda value: 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 4 if value == 'd' else 999)

#Categorical variables with non-integer labels mapped to integers
#Assortment (in store data set)
# df_store.Assortment.unique() = array(['a', 'c', 'b'], dtype=object)
df_store['Assortment'] = df_store['Assortment'].apply(lambda value: 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 999)

#Categorical variables with non-integer labels mapped to integers
#PromoInterval (in store data set)
#df_store.PromoInterval.unique() = array([nan, 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec'], dtype=object)
df_store['PromoInterval'] = df_store['PromoInterval'].apply(lambda value: 1 if value == 'Jan,Apr,Jul,Oct' else 2 if value == 'Feb,May,Aug,Nov' else 3 if value == 'Mar,Jun,Sept,Dec' else 999)

#- Transform Date - To a Year-month level or Year-Week level

print ('COMPLETE -- Process to convert non-integer labels to integers')

#Merge train nand store datasets into one master data set - df_master0
#URL: http://pandas.pydata.org/pandas-docs/stable/merging.html
df_master0 = pd.merge(df_train, df_store, how='left', on=['Store'])

#Write df_master0 to CSV
df_master0.to_csv('master0.csv', sep=',', index=False, encoding='utf-8')

