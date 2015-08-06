
import numpy as np
from .base import BaseEstimator, clone

def _get_point_scores(model, X, y):
    scores = [model.score(X[k:k+1], y[k:k+1]) for k in xrange(X.shape[0])]
    scores = np.array(scores)
    return scores

class RANSAC(BaseEstimator):
    def __init__(self, model, min_samples=3, max_trials=100, 
                 inlier_thresh=None, stop_n_inliers=100):
        self.base_estimator = model
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.inlier_thresh = inlier_thresh


    def fit(self, X, y=None):
        """
        Fit the RANSAC model to data
        
        Parameters
        ----------
        X : np.array, shape = [n_samples, n_features]
            Independent variables
        y : np.array or None, shape = [n_samples]
            Value to predict (or None for unsupervised
            methods)

        Returns
        -------
        self
        """
        supervised = True
        if y is None:
            supervised = False
            
        # should have some check statements here
        # to make sure X, and y are good
        n_samples, _ = X.shape

        for k in xrange(self.max_trials):
            this_estimator = clone(self.base_estimator)

            this_subset = np.random.permutation(np.arange(n_samples))[:self.min_samples]
            this_X = X[this_subset]
            if supervised:
                this_y = y[this_subset]
            else:
                this_y = None

            this_estimator.fit(this_X, this_y)           

            these_scores = _get_point_scores(this_estimator, X, y)
            this_score = these_scores[this_subset].sum()
            for i in xrange(self.max_iters):
                new_subset = np.where(these_scores < self.inlier_thresh)[0]
                if set(new_subset) == set(this_subset):
                    break

                new_X = X[new_subset]
                if supervised:
                    new_y = y[new_subset]
                else:
                    new_y = None
                
                this_estimator.fit(new_X, new_y)
                new_scores = _get_point_scores(this_estimator, X, y)
                new_score = new_scores[new_subset].sum()
                if these_scores[new_subset].sum() > new_scores[new_subset].sum():
                    this_score = new_scores[new_subset].sum()
                    these_scores = new_scores
                else:
                    break

            n_inliers = len(this_subset)
            if n_inliers >= self.stop_n_inliers:
                break

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(this_X, this_y)
        self.inlier_inds_ = this_subset

        return self
