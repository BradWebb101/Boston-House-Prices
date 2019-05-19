# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:16:06 2019

@author: bradw
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

#Loading Data
df_train = pd.read_csv('train.csv', index_col=[0])
df_test = pd.read_csv('test.csv', index_col=[0])

#Combining data sets for data wrangling
df_combine = pd.concat([df_train,df_test],sort=True)

#looking at data and ways to improve on the layout of the data
print(df_combine.isna().sum())

#Data wrangling 
#dropping features with high nan and little detail for regression
df_combine = df_combine.drop(['Alley','Fence', 'FireplaceQu','MiscFeature','PoolQC'],axis=1)

#Converting sparse data into binary variable
df_combine['Pool'] = df_combine['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_combine.drop('PoolArea',axis=1,inplace=True)
df_combine['EnclosedPorch'] = df_combine['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
df_combine['3SsnPorch'] = df_combine['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)

#Converting year details into previous year number to standadise data where data is numerically significant
df_combine['YearOld'] = df_combine['YearBuilt'].apply(lambda x: (x - 2019)*-1)
df_combine['YearsSinceRemodel'] = df_combine['YearRemodAdd'].apply(lambda x: (x-2019)*-1)
df_combine['YrSold'] = df_combine['YrSold'].astype('category')
df_combine['YrSold'] = df_combine['YrSold'].cat.codes
df_combine.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1,inplace=True)

#Creating dummy variables where data is not numerically significant eg sale condition (Partial, Normal ect)
df_combine = df_combine.fillna(0)

#Dropping sale price from features to create dummies (Add back in laters)
df_combine_saleprice = df_combine['SalePrice']
df_combine.drop('SalePrice',axis=1,inplace=True)

for columns in df_combine.select_dtypes(include='object').columns:
    dummy = pd.get_dummies(df_combine[columns],drop_first=True)
    dummy.rename(columns=lambda x: str([columns]) + str(x), inplace=True)
    df_combine = pd.concat([df_combine,dummy],axis=1)
    df_combine.drop([columns],axis=1,inplace=True)

df_combine = pd.concat([df_combine,df_combine_saleprice],axis=1)


df_train = df_combine[df_combine['SalePrice'] != 0]
df_test = df_combine[df_combine['SalePrice'] == 0]
df_test.drop('SalePrice',axis=1,inplace=True)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['SalePrice'],axis=1),df_train['SalePrice'])

#Choosing model
rr_model = Ridge(random_state=0).fit(X_train, y_train)
svr_linear_model = SVR(kernel="linear").fit(X_train, y_train)
svr_rbf_model = SVR(kernel="rbf").fit(X_train, y_train)
rfr_model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
gbr_model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
ada_model = AdaBoostRegressor().fit(X_train, y_train)

models = [rr_model, svr_linear_model, svr_rbf_model, rfr_model, gbr_model, ada_model]

#Scoring Models
chosen_model = None
highest_score = 0

for model in models:
    score = model.score(X_train, y_train)
    print(model)
    if score > highest_score:
        chosen_model = model
        highest_score = score
    
#Ridge is the highest so lets implement the model
predictions = pd.DataFrame(gbr_model.predict(df_test),index=df_test.index,columns=['SalePrice'])        
predictions.to_csv('Ridge Predictions.csv')

