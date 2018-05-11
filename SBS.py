
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    """Sequential Backward Selection algorithm.
    The algorithms removes variables ones after the others up to a certain 
    point, based on the highest score of the scoring method. 
    
    Parameters
    -----------------
    estimator: algorithm to use
    k_features: number of variables we want to keep
    scoring: way to estimate the result of the algorithm
    test_size: test size
    random_state: random state
    
    """
    
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        """Initialize"""
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """Run the algorithm"""
        
        # Split the data
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size, 
                                 random_state=self.random_state)
        # Indices
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        # Initialise the first score
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        # While the dimension is still above the point specified
        while dim > self.k_features:
            
            # Initialise results for this run
            scores = []
            subsets = []

            # Loop through all potentials combinations of n-1 variables
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # Rank based on the best score (i.e, which removed variables gives
            # the best score)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        """Computes the forecasting score, based on calibration on a part of 
        the sample and score computing in the other"""
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        
        
## ------------------------------------------------------
## Example
#from sklearn.neighbors import KNeighborsClassifier
#import matplotlib.pyplot as plt
#
#knn = KNeighborsClassifier(n_neighbors=2)
#
## selecting features
#sbs = SBS(knn, k_features=1)
#sbs.fit(X, y)
#
## plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]
#
#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0.7, 1.1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()
## ------------------------------------------------------