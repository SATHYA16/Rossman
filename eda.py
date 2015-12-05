# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:23:23 2015

@author: Sathya
"""
import pandas as pd

df = pd.read_csv('master0.csv',parse_dates=True)

df['Month'] = df['Date'].str[5:7]

s1 = df['Sales'].groupby(df['Month']).mean()
print s1

s2 = df['Sales'].groupby(df['DayOfWeek']).mean()
print s2

s3 = df['Sales'].groupby(df['StoreType']).mean()
print s3

#df['Day'] = df['Date'].str[8:]
s4 = df.groupby(['StoreType', 'DayOfWeek']).mean()['Sales']
print s4

s4 = df.groupby(['StateHoliday','StoreType']).mean()['Sales']
print s4

s4 = df.groupby(['Open','Assortment']).mean()['Sales']
print s4

#df['Date'] = pd.to_datetime(df['Date'])
#print df['Date'].dt.week()