# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:30:15 2015

@author: apple
"""

import re
import datetime
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.svm import SVR

########################################################################################
print ('############################### START ################################')

#Assign the absolute path of the data file to a variable
file_path_mastertrain = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656/Project/Rossman/mastertrain.csv')
file_path_mastertest = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656/Project/Rossman/mastertest.csv')

#Assign the contents of the datafile to a Pandas dataframe
#Indicate to treat first row as column names
df_mastertrain = pd.read_csv(file_path_mastertrain, sep=',', header=0)
df_mastertest = pd.read_csv(file_path_mastertest, sep=',', header=0)


#Build models and compare their properties in this part

#Declare dependent and independent variables
#independent_vars = ['Store', 'Open', 'Promo', 'SchoolHoliday', 'DoW1', 'DoW2', 'DoW3', 'DoW4', 'DoW5', 'DoW6', 'DoW7', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25', 'W26', 'W27', 'W28', 'W29','W30', 'W31', 'W32', 'W33', 'W34', 'W35', 'W36', 'W37', 'W38', 'W39', 'W40', 'W41', 'W42', 'W43', 'W44', 'W45', 'W46', 'W47', 'W48', 'W49', 'W50', 'W51', 'W52', 'Y2013', 'Y2014', 'Y2015', 'StateHolidayNO', 'StateHolidayA', 'StateHolidayB', 'StateHolidayC', 'CompetitionDistance', 'Promo2', 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 'AssortmentA', 'AssortmentB', 'AssortmentC', 'NoOfWeeksSinceCompetition', 'NoOfWeeksSincePromo2', 'PromoIntervalJan', 'PromoIntervalApr', 'PromoIntervalJul', 'PromoIntervalOct', 'PromoIntervalFeb', 'PromoIntervalMay', 'PromoIntervalAug', 'PromoIntervalNov', 'PromoIntervalMar', 'PromoIntervalJun', 'PromoIntervalSept', 'PromoIntervalDec']
#emoved: Store, 52 Weeks, 3 Years 
independent_vars = ['Open', 'Promo', 'SchoolHoliday', 'DoW1', 'DoW2', 'DoW3', 'DoW4', 'DoW5', 'DoW6', 'DoW7', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'W16', 'W17', 'W18', 'W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25', 'W26', 'W27', 'W28', 'W29','W30', 'W31', 'W32', 'W33', 'W34', 'W35', 'W36', 'W37', 'W38', 'W39', 'W40', 'W41', 'W42', 'W43', 'W44', 'W45', 'W46', 'W47', 'W48', 'W49', 'W50', 'W51', 'W52', 'StateHolidayNO', 'StateHolidayA', 'StateHolidayB', 'StateHolidayC','Sales_recent','customer_recent','Sales_total','customer_total','CompetitionDistance', 'Promo2', 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 'AssortmentA', 'AssortmentB', 'AssortmentC', 'NoOfWeeksSinceCompetition', 'NoOfWeeksSincePromo2', 'PromoIntervalJan', 'PromoIntervalApr', 'PromoIntervalJul', 'PromoIntervalOct', 'PromoIntervalFeb', 'PromoIntervalMay', 'PromoIntervalAug', 'PromoIntervalNov', 'PromoIntervalMar', 'PromoIntervalJun', 'PromoIntervalSept', 'PromoIntervalDec']
dependent_var = []
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
clf = SVR(kernel='rbf', C=1e3, gamma=0.1)

#Fit using model
clf.fit(df_mastertrain[independent_vars], df_mastertrain['Sales'])

#Predict using model
#clf.predict(df_mastertest[independent_vars])
df_result = pd.DataFrame(df_mastertest['Id'])
df_result['Sales'] = clf.predict(df_mastertest[independent_vars])
df_result.to_csv('result.csv', sep=',', index=False, columns = ['Id','Sales'], encoding='utf-8')
'''
'''
print('-----------------------------------')
'''
#PCA
print ('        PCA')


pca = decomposition.PCA()
pca.fit_transform(df_mastertrain[independent_vars])
print(pca.explained_variance_)
PCA(copy=True, n_components=None, whiten=False)

'''
'''
print('###################')
print(' Test Error Rate:  ')
print('###################')
accuracy_clf = metrics.zero_one_loss(df_mastertrain['Sales'], clf.predict(df_mastertest[independent_vars]))
print (1-accuracy_clf)
print('#########################')
print(' Classification Report:  ')
print('#########################')
print(metrics.classification_report(df_mastertrain['Sales'], clf.predict(df_mastertest[independent_vars])))
print('############')
print(' Crosstab:  ')
print('############')
print(pd.crosstab(df_mastertest['Sales'], clf.predict(df_mastertrain[independent_vars]), margins = True))
print('####################')
print(' Confusion Matrix:  ')
print('####################')
print(metrics.confusion_matrix(df_mastertrain['Sales'], clf.predict(df_mastertest[independent_vars])))
print('############')
print(' ROC Curve: ')
print('############')
false_positive_rate, true_positive_rate, thresholds = roc_curve(df_mastertrain['Sales'], clf.predict(df_mastertest[independent_vars]))
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic Curve')
plt.plot(false_positive_rate, true_positive_rate, 'g', label='AUC = %0.2f'% roc_auc)
plt.plot([0,1],[0,1],'--', label='Pure chance') #random probability, i.e., curve for model with 50-50 classification 
plt.legend(loc="lower right")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('#########################')
print(' Precision-Recall Curve: ')
print('#########################')
precision, recall, thresholds = precision_recall_curve(df_mastertrain['Sales'], clf.predict(df_mastertest[independent_vars]))
prc_auc = auc(recall, precision) #even the below line can be used to get the same result
#prc_auc = average_precision_score(test['class'], clf.predict(test[independent_vars]), average='macro', sample_weight=None)
plt.clf() #clear current figure
plt.plot(recall, precision, 'g', label='AUC = %0.2f'% prc_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()
'''
print ('############################### END ##################################\n')
