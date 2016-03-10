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
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from pandas.stats.api import ols
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

########################################################################################
print ('############################### START OF PART 1 ################################')

#Assign the absolute path of the data file to a variable
file_path_mastertrain = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656/Project/Rossman/mastertrain.csv')
file_path_mastertest = os.path.abspath('C:/Sathya/Subject/Fall-2015/BIA-656/Project/Rossman/mastertest.csv')

#Assign the contents of the datafile to a Pandas dataframe
#Indicate to treat first row as column names
df_mastertrain = pd.read_csv(file_path_mastertrain, sep=',', header=0)
df_mastertest = pd.read_csv(file_path_mastertest, sep=',', header=0)

print ('############################### END OF PART 1 ##################################\n')
########################################################################################

print ('############################### START OF PART 2 ################################')
#Build models and compare their properties in this part

#Declare dependent and independent variables
independent_vars = df_mastertrain[['Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'Promo2','StoreTypeB','AssortmentC','AssortmentA','DoW1','DoW3','DoW4','DoW7','StateHolidayB', 'NoOfWeeksSinceCompetition', 'NoOfWeeksSincePromo2','PromoIntervalJan', 'PromoIntervalApr', 'PromoIntervalJul', 'PromoIntervalOct', 'PromoIntervalFeb', 'PromoIntervalMay', 'PromoIntervalAug', 'PromoIntervalNov', 'PromoIntervalMar', 'PromoIntervalJun', 'PromoIntervalSept', 'PromoIntervalDec']]
dependent_var = df_mastertrain['Sales']
print('-----------------------------------')

reg = ols(y= dependent_var,x=independent_vars)
print reg

#rf = RandomForestRegressor() # initialize
#rf.fit(independent_vars,dependent_var)
#print 'R-squared for Random Forest:     ',rf.score(independent_vars,dependent_var)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf.fit(independent_vars,dependent_var)
svr_lin.fit(independent_vars,dependent_var)
svr_poly.fit(independent_vars,dependent_var)
print 'R-squared for SVR:     ',svr_rbf.score(independent_vars,dependent_var)
print 'R-squared for SVR:     ',svr_lin.score(independent_vars,dependent_var)
print 'R-squared for SVR:     ',svr_poly.score(independent_vars,dependent_var)

##CART
#print ('        CART Model')
##http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
#clf = DecisionTreeRegressor(random_state = 0)
#
##Fit using model
#clf.fit(df_mastertrain[independent_vars], df_mastertrain[dependent_var])
#
##Predict using model
#clf.predict(df_mastertest[independent_vars])
#
#df_result = pd.DataFrame(df_mastertest['Id'])
#
#df_result['Sales'] = clf.predict(df_mastertest[independent_vars])
#
#df_result.to_csv('result.csv', sep=',', index=False, columns = ['Id','Sales'], encoding='utf-8')
#
#print (export_graphviz(clf, out_file='tree.dot'))

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
print ('############################### END OF PART 2 ##################################\n')
