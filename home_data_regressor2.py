# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:09:38 2020

@author: admin
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#%% reading data

def get_data():
    full_train_data = pd.read_csv('train.csv', index_col = 'Id')
    full_test_data = pd.read_csv('test.csv', index_col = 'Id')
    return full_train_data, full_test_data

#%% preprocessing data
    
def clean_data(full_train_data, full_test_data):
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
    return X, y, X_valid

#%% data scaling
    
def scaled_data(X, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    X_valid = scaler_X.transform(X_valid)
    return X, X_valid, scaler_X

#%%
def performance(y, y_pred):
    metric = np.sqrt(mean_squared_error(y, y_pred))
    return metric    

#%% regressor functions
    
def regressor_fn_optimised(X, y, X_valid, choice):      
    from sklearn.metrics import make_scorer
    my_scorer = make_scorer(performance, greater_is_better = False)
    from bayes_opt import BayesianOptimization
    
    if choice == 1:    
        from sklearn.linear_model import Ridge        
        def regressor_fn(alpha):            
            regressor = Ridge(alpha = alpha)        
            cval = cross_val_score(regressor, X, y, scoring = my_scorer, cv = 5)
            return cval.mean()
        pbounds = {'alpha': (0, 1000)}
        
    elif choice == 2:    
        from sklearn.ensemble import RandomForestRegressor        
        def regressor_fn(n_estimators, max_depth):     
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)        
            cval = cross_val_score(regressor, X, y, scoring = my_scorer, cv = 5)
            return cval.mean()
        pbounds = {'n_estimators': (10, 500), 'max_depth': (2,20)}
        
    elif choice == 3: 
        X, X_valid, scaler_X = scaled_data(X, X_valid)
        from sklearn.svm import SVR        
        def regressor_fn(C, gamma):            
            regressor = SVR(C = C, kernel = 'sigmoid', gamma = gamma)        
            cval = cross_val_score(regressor, X, y, scoring = my_scorer, cv = 5)
            print(cval)
            return cval.mean()
        pbounds = {'C': (0.1, 1000), 'gamma': (0.001, 100)}
        
    elif choice == 4:
        from lightgbm.sklearn import LGBMRegressor
        def regressor_fn(learning_rate, max_depth):            
            max_depth = int(max_depth)
            regressor = LGBMRegressor(learning_rate = learning_rate, max_depth = max_depth)        
            cval = cross_val_score(regressor, X, y, scoring = my_scorer, cv = 5)
            return cval.mean()
        pbounds = {'learning_rate': (0.001, 1), 'max_depth': (2,20)}        
        
    else:
        from xgboost import XGBRegressor
        def regressor_fn(learning_rate, n_estimators, max_depth):            
            n_estimators, max_depth = int(n_estimators), int(max_depth)
            regressor = XGBRegressor(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth)        
            cval = cross_val_score(regressor, X, y, scoring = my_scorer, cv = 5)
            return cval.mean()
        pbounds = {'learning_rate': (0.001, 1), 'n_estimators': (10, 5000), 'max_depth': (2,20)}
    
    optimizer = BayesianOptimization(regressor_fn, pbounds, verbose = 2)
    optimizer.maximize(init_points = 5, n_iter = 50)
    # change next line in accordance with choice of regressor made
    y_valid_pred = XGBRegressor(learning_rate = optimizer.max['params']['learning_rate'], n_estimators = int(optimizer.max['params']['n_estimators']), max_depth = int(optimizer.max['params']['max_depth'])).fit(X, y).predict(X_valid)
    return y_valid_pred, optimizer.max

#%% main
    
if __name__ == '__main__':
    full_train_data, full_test_data = get_data()
    X, y, X_valid = clean_data(full_train_data, full_test_data)
    y_valid_pred, optimal_params = regressor_fn_optimised(X, y, X_valid, choice = 5)      
    df = pd.DataFrame({'Id':X_valid.index, 'SalePrice':y_valid_pred})
    df.to_csv('prediction.csv', index = False)
    
    