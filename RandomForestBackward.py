# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:21:48 2018

@author: pages
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class RandomForestBackward():
    """Random Forest backward algorithm, remove variables based on the backtest
    contribution.
    The algo randomly sample a part of the measures to computes a backtest and
    rank variables by importance. Then it tries to remove them, considering the 
    least important
    
    Attributes
    -----------------  
    - n_iter: maximum number of iterations. Note that in any case, once passed
    through all the variables, if there is no improvement, the process stops
    - n_estimators = number of trees in the Random Forest
    - sample_size: percentage of data to consider for random sampling
    - n_folds: Number of folds used for cross validation
    """
    
    def __init__(self, n_iter = 100, 
                 n_estimators = 20, 
                 sample_size = 0.9, 
                 n_folds = 5):
        """Initialize the algorithm"""
        
        self.n_iter = n_iter
        self.n_estimators = n_estimators
        self.sample_size =sample_size
        self.n_folds = n_folds
        self._state = True
        
    def fit(self, X, y):
        """Remove variables based on their contribution to the backtest"""
        
        # Initialize classifier and algorithm errors
        r = RandomForestRegressor(n_estimators = self.n_estimators,
                                  n_jobs = -1)
        self._error = []
        self._removed_variables = []
        
        # Initial point
        error, features = self.step(X, y, r)
        self._error.append(error)
        
        while self._state:
            
            # Initialize the algo with a wrong state
            self._state = False
            incr = 0
            
            # Max number of iterations
            for _ in range(self.n_iter):
                
                # Need to still have a input matrix
                if X.shape[1] >= 1:
                
                    # If we reach the point where all variables have been tested
                    if incr == len(features):
                        print('There is no more operation to realize')
                        break
                    
                    # Use of incremental variable to iteratively test the variables
                    X_int = self.remove_variable(X, features[incr])
                    
                    error, features_int = self.step(X_int, y, r)
                    
                    if error <= self._error[-1]:
                        self._error.append(error)
                        self._state = True
                        self._removed_variables.append(features[incr])
                        X = self.remove_variable(X, features[incr])
                        features = features_int
                        incr = 0
                        break
                    else:
                        incr += 1
                    
    def my_error(self, X, y, r):
        """Compute the cross validation error"""
        
        kf = KFold(n_splits = self.n_folds, shuffle = True, random_state = 42)
        
        # Initialize results
        r_res = np.zeros(X.shape[0])
        
        # Backtest
        for train, test in kf.split(X):
            x_calib = X.iloc[train,:]
            y_calib = y[train]
            x_valid = X.iloc[test,:]
            
            # Fit models
            r.fit(x_calib, y_calib.reshape(-1,))
            
            # Fill variables to be compared
            r_res[test] = r.predict(x_valid)
            
        # Return metrics
        return mean_absolute_error(y, r_res)
    
    def random_sampling(self, X, y):
        """Return randomly sampled variables, bringing some noise in the 
        approach"""
        
        perm = np.random.permutation(len(y)).reshape(-1,1)[:int(len(y) * self.sample_size)].reshape(-1,)

        return X.iloc[perm,:], y[perm]
    
    def step(self, X, y, r):
        """Individual iteration"""
        
        X_r, y_r = self.random_sampling(X, y)
        return self.my_error(X_r, y_r, r), self.features(X_r, y_r, r)
    
    def features(self, X, y, r):
        """Return the features ranked by importance"""
        
        r.fit(X, y.reshape(-1,))
        return list(X.columns[np.argsort(r.feature_importances_)])
    
    def remove_variable(self, X, var):
        """Return dataset with one less variable"""
        
        # List of variables
        var_list = list(X.columns)
        var_list.remove(var)
        
        return X[var_list]
    
    
        
        
        
        
        
        