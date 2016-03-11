# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:30:15 2015

@author: SUBRAMANIAN VELLAIYAN

"""

#import re
#import datetime
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO 
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import decomposition

path = 'C:/Sathya/Subject/Fall-2015/BIA-656/Project/Rossman/'
#If this is 'train' then the rest of the code will run on train, if it is 'test', it will work on test file.
dataset = ['train','test']

for data in dataset: 
    #Assign the absolute path of the data file to a variable
    file_path_train = os.path.abspath(path+'{0}.csv'.format(data))
    file_path_store = os.path.abspath(path+'store.csv')
    
    #Assign the contents of the datafile to a Pandas dataframe
    df_train = pd.read_csv(file_path_train, sep=',', header=0) #Indicate to treat first row as column names
    df_store = pd.read_csv(file_path_store, sep=',', header=0) #Indicate to treat first row as column names
    
    ##########################################################################################################
    
    print ('START -- Process to convert non-integer labels to integers')
    
    #Categorical variables with non-integer labels mapped to integers
    #StateHoliday (in train data set)
    #df_train.StateHoliday.unique() = array(['0', 'a', 'b', 'c', 0], dtype=object)
    #0 = ('0' or 0) 1 = 'a' 2 = 'b' 3 = 'c' else 999
    df_train['StateHoliday'] = df_train['StateHoliday'].apply(lambda value: 0 if (value == '0' or value == 0) else 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 999)
    
    #Categorical variables with non-integer labels mapped to integers
    #StoreType (in store data set)
    #df_store.StoreType.unique() = array(['c', 'a', 'd', 'b'], dtype=object)
    #1 = 'a' 2 = 'b' 3 = 'c' 4 = 'd' else 999
    df_store['StoreType'] = df_store['StoreType'].apply(lambda value: 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 4 if value == 'd' else 999)
    
    #Categorical variables with non-integer labels mapped to integers
    #Assortment (in store data set)
    #df_store.Assortment.unique() = array(['a', 'c', 'b'], dtype=object)
    #1 = 'a' 2 = 'b' 3 =='c' else 999
    df_store['Assortment'] = df_store['Assortment'].apply(lambda value: 1 if value == 'a' else 2 if value == 'b' else 3 if value == 'c' else 999)
    
    #Categorical variables with non-integer labels mapped to integers
    #PromoInterval (in store data set)
    #df_store.PromoInterval.unique() = array([nan, 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec'], dtype=object)
    #1 = 'Jan,Apr,Jul,Oct' 2 = 'Feb,May,Aug,Nov' 3 = 'Mar,Jun,Sept,Dec' else 999
    
    df_store['PromoInterval'] = df_store['PromoInterval'].apply(lambda value: 1 if value == 'Jan,Apr,Jul,Oct' else 2 if value == 'Feb,May,Aug,Nov' else 3 if value == 'Mar,Jun,Sept,Dec' else 999)
    
    print ('COMPLETE -- Process to convert non-integer labels to integers')
    
    ##########################################################################################################
    #The property of scikit is such that it does not treat categorical variables as categories
    #Instead, it will treat integers and floats are ordinal and convert strings into floats
    #This results in gibberish model fitting. Therefore it is important to split each categorical variable into multiple columns, with only two categories '1' and '0'
    #The following section of the code does this process
    
    #Categorical variable 'DayOfWeek' (in train data set)
    #It has seven uniques values: 1 to 7. Creating 7 columns with 1/0 to mean Yes/No
    #New Columns: DoW1, DoW2, DoW3, DoW4, DoW5, DoW6, DoW7
    
    df_train['DoW1'] = np.where(df_train['DayOfWeek'] == 1, 1, 0) # Create a new column, impute 1 or 0 based on current value
    df_train['DoW2'] = np.where(df_train['DayOfWeek'] == 2, 1, 0)
    df_train['DoW3'] = np.where(df_train['DayOfWeek'] == 3, 1, 0)
    df_train['DoW4'] = np.where(df_train['DayOfWeek'] == 4, 1, 0)
    df_train['DoW5'] = np.where(df_train['DayOfWeek'] == 5, 1, 0)
    df_train['DoW6'] = np.where(df_train['DayOfWeek'] == 6, 1, 0)
    df_train['DoW7'] = np.where(df_train['DayOfWeek'] == 7, 1, 0)
    
    del df_train['DayOfWeek']
    
    #Categorical variable 'Date' (in train data set)
    #This column is split into 53 columns for weeks and 3 year columns with 1/0 to mean Yes/No
    #New Columns: W1, W2, W3,... W53 and Y1, Y2, and Y3
    
    df_train['Date_Mod'] = pd.to_datetime(df_train['Date']) #to_datetime(argument): Convert argument to datetime
    df_train['Week'] = df_train['Date_Mod'].dt.week #create week column that gives 1-52 values based on week of year
    df_train['Year'] = df_train['Date_Mod'].dt.year #create year column that gives 1-3 values based on week of year
    
    #Create 52 columns (for weeks) and fill with appropriate values
    for i in range(1,53):
        df_train['W{0}'.format(i)] = (df_train['Week'] == i).astype('int')
        
    #Create 3 columns (for years) and fill with appropriate values
    for i in range(2013,2016):
        df_train['Y{0}'.format(i)] = (df_train['Year'] == i).astype('int')
        
    del df_train['Date']
    #del df_train['Date_Mod'] #This is used later to modify another column, so not deleting for now
    del df_train['Week']
    del df_train['Year']
    
    #Categorical variable 'StoreType' (in store data set)
    #This column is split into 4 columns for the 4 types of stores with 1/0 to mean Yes/No
    
    df_store['StoreTypeA'] = (df_store['StoreType'] == 1).astype('int')
    df_store['StoreTypeB'] = (df_store['StoreType'] == 2).astype('int')
    df_store['StoreTypeC'] = (df_store['StoreType'] == 3).astype('int')
    df_store['StoreTypeD'] = (df_store['StoreType'] == 4).astype('int')
    
    del df_store['StoreType']
    
    #Categorical variable 'Assortment' (in store data set)
    #This column is split into 3 columns for the 3 types of stores with 1/0 to mean Yes/No
    
    df_store['AssortmentA'] = (df_store['Assortment'] == 1).astype('int')
    df_store['AssortmentB'] = (df_store['Assortment'] == 2).astype('int')
    df_store['AssortmentC'] = (df_store['Assortment'] == 3).astype('int')
    
    del df_store['Assortment']
    
    #Categorical variable 'StateHoliday' (in store data set)
    #This column is split into 4 columns for the 4 types of stores with 1/0 to mean Yes/No
    
    df_train['StateHolidayNO'] = (df_train['StateHoliday'] == 0).astype('int')
    df_train['StateHolidayA'] = (df_train['StateHoliday'] == 1).astype('int')
    df_train['StateHolidayB'] = (df_train['StateHoliday'] == 2).astype('int')
    df_train['StateHolidayC'] = (df_train['StateHoliday'] == 3).astype('int')
    
    del df_train['StateHoliday']
    
    #Imputing missing values in the store dataset column 'CompetitionDistance' with 0
    df_store['CompetitionDistance'] = df_store['CompetitionDistance'].fillna(0)
    
    #Columns with missing values
    # -> CompetitionDistance
    # -> CompetitionOpenSinceMonth
    # -> CompetitionOpenSinceYear
    # -> Promo2SinceWeek 
    # -> Promo2SinceYear
    #Should we replace all missing values with 999? Not for now.
    
    #Merge train nand store datasets into one master data set - df_master0
    #URL: http://pandas.pydata.org/pandas-docs/stable/merging.html
    df_master0 = pd.merge(df_train, df_store, how='left', on=['Store'])
    
    #Creating column for CompetitionOpenSinceMonth	& CompetitionOpenSinceYear - 'NoOfMonthsSinceCompetition'
    #either values exist in both columns or none in either
    df_master0['NoOfWeeksSinceCompetition'] = ((df_master0['Date_Mod'].dt.year - df_master0['CompetitionOpenSinceYear']) * 12 + df_master0['Date_Mod'].dt.month - df_master0['CompetitionOpenSinceMonth'])
    df_master0['NoOfWeeksSinceCompetition'] = df_master0['NoOfWeeksSinceCompetition'].fillna(0) 
    df_master0['NoOfWeeksSinceCompetition'] = df_master0['NoOfWeeksSinceCompetition'].apply(lambda value: 0 if value < 0 else value*4)
    
    del df_master0['CompetitionOpenSinceMonth']
    del df_master0['CompetitionOpenSinceYear']
    
    #Creating column for Promo2SinceWeek &	Promo2SinceYear - 'NoOfWeeksSincePromo2'
    #either values exist in both columns or none in either
    df_master0['NoOfWeeksSincePromo2'] = ((df_master0['Date_Mod'].dt.year - df_master0['Promo2SinceYear']) * 52 + df_master0['Date_Mod'].dt.week - df_master0['Promo2SinceWeek'])
    df_master0['NoOfWeeksSincePromo2'] = df_master0['NoOfWeeksSincePromo2'].fillna(0) 
    df_master0['NoOfWeeksSincePromo2'] = df_master0['NoOfWeeksSincePromo2'].apply(lambda value: 0 if value < 0 else value)
    
    del df_master0['Promo2SinceWeek']
    del df_master0['Promo2SinceYear']
    
    #Create 12 columns (for months) and fill with appropriate values (1 if Promo2 is in this interval 0 otherwise)
    
    for i in range(1,4):
        df_master0['PromoIntervalM{0}'.format(i)] = (df_master0['PromoInterval'] == i).astype('int')
    
    df_master0['PromoIntervalJan'] = df_master0['PromoIntervalM1']
    df_master0['PromoIntervalApr'] = df_master0['PromoIntervalM1']
    df_master0['PromoIntervalJul'] = df_master0['PromoIntervalM1']
    df_master0['PromoIntervalOct'] = df_master0['PromoIntervalM1']
    
    df_master0['PromoIntervalFeb'] = df_master0['PromoIntervalM2']
    df_master0['PromoIntervalMay'] = df_master0['PromoIntervalM2']
    df_master0['PromoIntervalAug'] = df_master0['PromoIntervalM2']
    df_master0['PromoIntervalNov'] = df_master0['PromoIntervalM2']
    
    df_master0['PromoIntervalMar'] = df_master0['PromoIntervalM3']
    df_master0['PromoIntervalJun'] = df_master0['PromoIntervalM3']
    df_master0['PromoIntervalSept'] = df_master0['PromoIntervalM3']
    df_master0['PromoIntervalDec'] = df_master0['PromoIntervalM3']
    
    del df_master0['PromoInterval']
    del df_master0['PromoIntervalM1']
    del df_master0['PromoIntervalM2']
    del df_master0['PromoIntervalM3']
    
    del df_master0['Date_Mod']
    
    df_master0['Open'] = df_master0['Open'].fillna(0)
    
    #Write df_master0 to CSV
    df_master0.to_csv('master{0}.csv'.format(data), sep=',', index=False, encoding='utf-8')
    
    
    
    
    '''
    unique_stores = list()
    unique_stores = df_master0.CompetitionDistance.unique()
    
    s2 = df_master0['Sales'].groupby(df_master0['Date']).sum()
    '''
    '''
    
    #Create a dictionary of what the variables are and what there description is
    master_dict = {
                    "Id": "This represents an identifier for a store on a given particular date.", 
                    "Store": "a unique Id for each store. This will be a categorical variable. There are 1115 unique Ids, representing the same number of stores.",
                    "Sales": "the turnover for any given day (Target Variable).",
                    "Customers": "the number of customers on a given day",
                    "Open": "an indicator for whether the store was open: Levels: 2 (0 = closed, 1 = open)",
                    "StateHoliday": "indicates a state holiday: Levels: 4 (a = public holiday, b = Easter holiday, c = Christmas, 0 = None \n Variables after mapping: 0 = ('0' or 0) 1 = 'a' 2 = 'b' 3 = 'c' else 999)",
                    "SchoolHoliday": "indicates if a Store on a given Date was affected by the closure of public schools: Levels: 2 (0 = Closed, 1 = Open)",
                    "StoreType": "differentiates between 4 different store models: Levels: 4 (a, b, c, d) \n 1 = 'a' 2 = 'b' 3 = 'c' 4 = 'd' else 999",
                    "Assortment": "describes an assortment level: Levels: 3 (a = basic, b = extra, c = extended) \n 1 = 'a' 2 = 'b' 3 =='c' else 999",
                    "CompetitionDistance": "distance in meters to the nearest competitor store",
                    "CompetitionOpenSinceMonth": "gives the approximate month of the time the nearest competitor was opened",
                    "CompetitionOpenSinceYear": "gives the year of the time the nearest competitor was opened",                
                    "Promo": "indicates whether a store is running a promo on that day",
                    "Promo2": "Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating",
                    "Promo2SinceWeek": "describes the calendar week when the store started participating in Promo2",
                    "Promo2SinceYear": "describes the calendar year when the store started participating in Promo2",
                    "PromoInterval": "describes the consecutive intervals when Promo2 is started, naming the months the promotion is started anew. E.g. 'Feb,May,Aug,Nov' means each round starts in February, May, August, November of any given year for that store \n 1 = 'Jan,Apr,Jul,Oct' 2 = 'Feb,May,Aug,Nov' 3 = 'Mar,Jun,Sept,Dec' else 999",          
                    "DayOfWeek": "= array([5, 4, 3, 2, 1, 7, 6], dtype=int64)",
                    "Date": "YOU KNOW WHAT IT IS"
                }
    
    #Iterate through each column in the dataset and ouput high-level analysis results for each
    for number,each_column in enumerate(df_master0.columns.values):
        print ('-----------------------------------')    
        print ('     '+str(number+1)+'. '+each_column)
        #print ('About this variable: ' + master_dict[each_column])
        print ('Basic information about this variable:')
        print ('')    
        print (df_master0[each_column].describe())
        print ('Number of uniques: '+str(len(df_master0[each_column].unique())))
        #Output a barchart of frequencies for each category in all the variables in the dataset
        i = df_master0[each_column].unique() #array of all uniques in the variable
        j=list()
        for each_unique in df_master0[each_column].unique():
            j = j + [len(df_master0[df_master0[each_column] == each_unique])] #list of counts corresponding to all uniques in the variable
        #Plot Categories (uniques) vs. respective counts for each variable
        sns.axlabel("Categories","Count")
        plt.title(each_column)
        sns.barplot(i,j)
        plt.show()
    
    #Plot paiwise plot for all variables   
    #sns.pairplot(df_master[['income', 'firstDate', 'lastDate', 'amount', 'freqSales', 'saleSizeCode', 'starCustomer', 'lastSale', 'avrSale', 'class']],hue="class")
    #plt.show()
    
    Corelation matrix among all variables in the data set
    '''
    
    if data == 'train':
        print (df_master0.corr()['Sales'])
    else:
        print (df_master0.info())

########################################################################################
print ('############################### START ################################')

#Assign the absolute path of the data file to a variable
file_path_mastertrain = os.path.abspath(path+'mastertrain.csv')
file_path_mastertest = os.path.abspath(path+'mastertest.csv')

#Assign the contents of the datafile to a Pandas dataframe
#Indicate to treat first row as column names
df_mastertrain = pd.read_csv(file_path_mastertrain, sep=',', header=0)
df_mastertest = pd.read_csv(file_path_mastertest, sep=',', header=0)


#Build models and compare their properties in this part

#Declare dependent and independent variables
#independent_vars = ['Store', 'Open', 'Promo', 'SchoolHoliday', 'DoW1', 'DoW2', 'DoW3', 'DoW4', 'DoW5', 'DoW6', 'DoW7', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25', 'W26', 'W27', 'W28', 'W29','W30', 'W31', 'W32', 'W33', 'W34', 'W35', 'W36', 'W37', 'W38', 'W39', 'W40', 'W41', 'W42', 'W43', 'W44', 'W45', 'W46', 'W47', 'W48', 'W49', 'W50', 'W51', 'W52', 'Y2013', 'Y2014', 'Y2015', 'StateHolidayNO', 'StateHolidayA', 'StateHolidayB', 'StateHolidayC', 'CompetitionDistance', 'Promo2', 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 'AssortmentA', 'AssortmentB', 'AssortmentC', 'NoOfWeeksSinceCompetition', 'NoOfWeeksSincePromo2', 'PromoIntervalJan', 'PromoIntervalApr', 'PromoIntervalJul', 'PromoIntervalOct', 'PromoIntervalFeb', 'PromoIntervalMay', 'PromoIntervalAug', 'PromoIntervalNov', 'PromoIntervalMar', 'PromoIntervalJun', 'PromoIntervalSept', 'PromoIntervalDec']
#emoved: Store, 52 Weeks, 3 Years 
independent_vars = ['Open', 'Promo', 'SchoolHoliday', 'DoW1', 'DoW2', 'DoW3', 'DoW4', 'DoW5', 'DoW6', 'DoW7', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25', 'W26', 'W27', 'W28', 'W29','W30', 'W31', 'W32', 'W33', 'W34', 'W35', 'W36', 'W37', 'W38', 'W39', 'W40', 'W41', 'W42', 'W43', 'W44', 'W45', 'W46', 'W47', 'W48', 'W49', 'W50', 'W51', 'W52', 'StateHolidayNO', 'StateHolidayA', 'StateHolidayB', 'StateHolidayC', 'CompetitionDistance', 'Promo2', 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 'AssortmentA', 'AssortmentB', 'AssortmentC', 'NoOfWeeksSinceCompetition', 'NoOfWeeksSincePromo2', 'PromoIntervalJan', 'PromoIntervalApr', 'PromoIntervalJul', 'PromoIntervalOct', 'PromoIntervalFeb', 'PromoIntervalMay', 'PromoIntervalAug', 'PromoIntervalNov', 'PromoIntervalMar', 'PromoIntervalJun', 'PromoIntervalSept', 'PromoIntervalDec']
dependent_var = ['Sales']
print('-----------------------------------')

'''
#CART
print ('        CART Model')
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
clf = DecisionTreeRegressor(random_state = 0)

#Fit using model
clf.fit(df_mastertrain[independent_vars], df_mastertrain[dependent_var])

#Predict using model
clf.predict(df_mastertest[independent_vars])
df_result = pd.DataFrame(df_mastertest['Id'])
df_result['Sales'] = clf.predict(df_mastertest[independent_vars])
df_result.to_csv('result.csv', sep=',', index=False, columns = ['Id','Sales'], encoding='utf-8')
'''

#RandomForestRegressor
print ('RandomForestRegressor Model')
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
clf = RandomForestRegressor(n_estimators = 20)

#Fit using model
clf.fit(df_mastertrain[independent_vars], df_mastertrain[dependent_var])

#Predict using model
clf.predict(df_mastertest[independent_vars])
df_result = pd.DataFrame(df_mastertest['Id'])
df_result['Sales'] = clf.predict(df_mastertest[independent_vars])
df_result.to_csv('result.csv', sep=',', index=False, columns = ['Id','Sales'], encoding='utf-8')
'''
'''
print('-----------------------------------')
'''
print ('############################### END ##################################\n')

