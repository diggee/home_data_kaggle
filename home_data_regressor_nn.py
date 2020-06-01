# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:44:28 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:37:06 2020

@author: diggee
"""

#%% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

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
    
    # for column in cols_to_transform:
    #     plt.figure()
    #     sns.boxplot(x = column, y = 'SalePrice', data = full_train_data)
        
    classes = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
    full_train_data.replace(classes, inplace = True)
    full_test_data.replace(classes, inplace = True)
    full_train_data = full_train_data.select_dtypes(exclude = ['object'])
    full_test_data = full_test_data.select_dtypes(exclude = ['object'])
    
    X = full_train_data.drop('SalePrice', axis = 1)
    y = full_train_data['SalePrice']
    X_test = full_test_data
    return X, y, X_test

#%% data scaling
    
def scaled_data(X, X_test):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    return X, X_test, scaler

#%%
def performance(y, y_pred):
    metric = np.sqrt(mean_squared_error(y, y_pred))
    return metric    

#%% NN model
    
def neural_network(n_features, X, y):    

    model = Sequential()
    model.add(Dense(n_features*2, 'relu', kernel_initializer = 'glorot_normal', input_shape = (n_features, )))
    model.add(Dense(n_features, 'relu', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2'))  
    model.add(Dense(int(n_features/2), 'relu', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2'))
    model.add(Dense(int(n_features/4), 'relu', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2')) 
    model.add(Dense(int(n_features/8), 'relu', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2')) 
    model.add(Dense(int(n_features/16), 'relu', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2')) 
    model.add(Dense(1, kernel_initializer = 'glorot_normal'))
    
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 1000)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 20, min_lr = 0.001)
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    
    model.compile('adam', loss = 'mae', metrics = [tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    history = model.fit(X, y, epochs = 50000, verbose = 2, validation_split = 0.2, callbacks = [checkpoint, reduce_lr, early_stop])
    return model, history

#%% make plots
    
def make_plots(history):
    
    plt.figure()
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.legend()
    
    plt.figure()
    plt.plot(history.history['root_mean_squared_error'], label = 'RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label = 'val_RMSE')
    plt.legend()      

#%% main

if __name__ == '__main__':
    
    full_train_data, full_test_data = get_data()
    X, y, X_test = clean_data(full_train_data, full_test_data)
    X, X_test, scaler = scaled_data(X, X_test) 
    model, history = neural_network(X.shape[1], X, y.values)
    make_plots(history)
    
    model.load_weights('model.h5')
    predictions = model.predict(X_test)
    submission = pd.read_csv('sample_submission.csv')
    submission.SalePrice = predictions
    submission.to_csv('sample_submission.csv', index = False)    