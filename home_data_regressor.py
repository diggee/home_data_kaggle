# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:28:40 2020

@author: diggee
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% reading the data files

full_train_data = pd.read_csv('train.csv', index_col = 'Id')
full_test_data = pd.read_csv('test.csv', index_col = 'Id')

#%% preprocessing data

# dropping columns with almost all NaN values and columns that won't have an impact on model
cols_to_drop = ['MSSubClass','Neighborhood','Condition1','Condition2','BldgType','MSZoning','Alley','PoolQC',
                'MiscFeature','Fence','FireplaceQu','MiscVal','SaleType','SaleCondition','GarageType','Functional',
                'Heating','Electrical','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','BldgType','HouseStyle',
                'LotConfig','LandContour']
full_train_data.drop(cols_to_drop, axis = 1, inplace = True)
full_test_data.drop(cols_to_drop, axis = 1, inplace = True)

# dealing with training missing values
total_elements = full_train_data.shape[0]
for column in full_train_data.columns[full_train_data.isnull().any()]:
    if full_train_data[column].isnull().sum()/total_elements < 0.05:
        full_train_data.dropna(subset = [column], inplace = True)
    elif full_train_data[column].isnull().sum()/total_elements > 0.05 and full_train_data[column].dtype == int:
        full_train_data[column].fillna(full_train_data[column].mean(), inplace = True)
    else:
        full_train_data[column].fillna(full_train_data[column].value_counts().index[0], inplace = True)

# dealing with test missing values
total_elements = full_test_data.shape[0]
for column in full_test_data.columns[full_test_data.isnull().any()]:
    if full_test_data[column].dtype == int:
        full_test_data[column].fillna(full_test_data[column].mean(), inplace = True)
    else:
        full_test_data[column].fillna(full_test_data[column].value_counts().index[0], inplace = True)
        
# dealing with categorical values
categorical_cols = full_train_data.columns[full_train_data.dtypes == object]
cols_to_transform = []
for column in categorical_cols:
    if full_train_data[column].value_counts()[0]/full_train_data[column].value_counts().sum() > 0.9:
        full_train_data.drop(column, axis = 1, inplace = True)
        full_test_data.drop(column, axis = 1, inplace = True)          
    else:
        cols_to_transform.append(column)

for column in cols_to_transform:
    plt.figure()
    sns.boxplot(x = column, y = 'SalePrice', data = full_train_data)
    
classes = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
full_train_data.replace(classes, inplace = True)
full_test_data.replace(classes, inplace = True)
full_train_data = full_train_data.select_dtypes(exclude = ['object'])
full_test_data = full_test_data.select_dtypes(exclude = ['object'])

X = full_train_data.drop('SalePrice', axis = 1)
y = full_train_data['SalePrice']
X_valid = full_test_data
    
#%% regression model

def grid_search(parameters, regressor, criteria, X_train, y_train):
    from sklearn. model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = criteria, cv = 5)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print(best_parameters)
    return best_accuracy, best_parameters

def scaled_data(X_train, X_test, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_valid = scaler_X.transform(X_valid)
    return X_train, X_test, X_valid, scaler_X

def regression(X_train, X_test, X_valid, y_train, choice):
    # reg_models = {1:'linear', 2:'polynomial', 3:'decision tree', 4:'random forest', 5:'SVR', 6:'XG Boost'}
    if choice == 1:
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()

    elif choice == 2:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        parameters = {'degree':[2, 3, 4, 5, 6]}
        _, best_param = grid_search(parameters, PolynomialFeatures(), 'neg_mean_squared_error', X_train, y_train)
        poly_reg = PolynomialFeatures(degree = best_param['degree'])
        X_poly = poly_reg.fit_transform(X_train)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y_train)
        regressor = lin_reg()

    elif choice == 3:
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor()

    elif choice == 4:
        from sklearn.ensemble import RandomForestRegressor
        parameters = {'n_estimators':[10, 50, 100, 200, 400, 500]}
        _, best_param = grid_search(parameters, RandomForestRegressor(), 'neg_mean_squared_error', X_train, y_train)
        regressor = RandomForestRegressor(n_estimators = best_param['n_estimators'])

    elif choice == 5:
        from sklearn.svm import SVR
        X_train, X_test, X_valid, scaler_X = scaled_data(X_train, X_test, X_valid)
        parameters = [{'C': [100, 200, 300, 400, 500], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.01, 0.1, 1]}]
        _, best_param = grid_search(parameters, SVR(), 'neg_mean_squared_error', X_train, y_train)
        regressor = SVR(kernel = best_param['kernel'], C = best_param['C'], gamma = best_param['gamma'])

    else:
        from xgboost import XGBRegressor
        parameters = [{'n_estimators':[500, 1000, 2000, 4000], 'learning_rate':[0.005, 0.01, 0.05, 1]}]
        _, best_param = grid_search(parameters, XGBRegressor(), 'neg_mean_squared_error', X_train, y_train)
        regressor = XGBRegressor(n_estimators = best_param['n_estimators'], learning_rate = best_param['learning_rate'])

    y_pred = regressor.fit(X_train, y_train).predict(X_test)
    return y_pred, regressor, X_valid

#%% evaluating prediction quality

def pred_quality(y_pred, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

#%% evalating the model on the test data and exporting it to csv file
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01 if X.shape[0]<=100000 else 10000)
y_pred, regressor, X_valid = regression(X_train, X_test, X_valid, y_train, choice = 4)
rmse, mae = pred_quality(y_pred, y_test)
print(mae)
      
y_valid_pred = regressor.predict(X_valid)
ID = np.arange(1461, 1461+len(y_valid_pred))
df = pd.DataFrame({'Id':ID, 'SalePrice':y_valid_pred})
df.to_csv('prediction.csv', index = False)
    