
import numpy as np
from .base import BaseEstimator, clone

def _get_point_scores(model, X, y=None):
    if y is None:
        scores = [model.score(X[k:k + 1]) for k in xrange(X.shape[0])]
    else:
        scores = [model.score(X[k:k + 1], y[k:k + 1]) for k in xrange(X.shape[0])]

    scores = np.array(scores)

    return scores

class RANSAC(BaseEstimator):
    """
    The RANSAC paradigm is very simple, and not very
    elegant, but it's useful for excluding outliers 
    contained in some dataset. 

    The fitting procedure works like this:

    1) Randomly sample m points from the dataset
    2) Fit the model to the m points
    3) Create a new subset of points that are consistent
        with this model
    4) If the size of this subset is big enough, then
        quit, otherwise go to (2) with this subset
    5) If this procedure hits a dead end then go to
        (1) and resample m points

    Parameters
    ----------
    model : sklearn.BaseEstimator instance
        Model to fit to some data. Can be any estimator.
    min_samples : int, optional
        Number of samples to construct the initial
        subset
    max_trials : int, optional
        Maximum number of trials before giving up
    inlier_thresh : float, optional
        The threshold to say whether a point is consistent
        with a model
    stop_n_inliers : int, optional
        Quit successfully whenever the subset contians 
        this many points
    point_scorer : callable, optional
        Function to compute the score of each point, which
        should correspond to how well the point is 
        explained by the model (higher score means more
        consistent with the model).
    """
    def __init__(self, model, min_samples=3, max_trials=100, 
                 inlier_thresh=np.log(0.05), stop_n_inliers=100,
                 point_scorer=_get_point_scores, min_score=-np.inf):
        self.base_estimator = model
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.inlier_thresh = inlier_thresh
        self.stop_n_inliers = stop_n_inliers
        self.point_scorer = point_scorer
        self.min_score = min_score


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
            this_estimator.best_score_ = -np.inf
            this_subset = np.random.permutation(np.arange(n_samples))[:self.min_samples]
            this_X = X[this_subset]
            if supervised:
                this_y = y[this_subset]
            else:
                this_y = None

            for i in xrange(self.max_trials):
                print 'Trial %d.%d (%d inliers / %d points | %2d%%) [%f]' % (k, i, len(this_subset), n_samples, len(this_subset) / float(n_samples) * 100, this_estimator.best_score_)

                this_estimator.fit(this_X, this_y)

                scores = self.point_scorer(this_estimator, X, y)

                print scores.min(), scores.mean(), scores.max()

                new_subset = np.where(scores > self.inlier_thresh)[0]

                if set(new_subset) == set(this_subset) or len(new_subset) == 0:
                    # we are stuck
                    break
                #if len(new_subset) <= len(this_subset):
                #    # we didn't get any better, so GTFO!
                #    break

                this_subset = new_subset
                this_X = X[this_subset]
                if supervised:
                    this_y = y[this_subset]
                else:
                    this_y = None

                if this_estimator.best_score_ < self.min_score:
                    continue

                if len(this_subset) >= self.stop_n_inliers:
                    break

            if this_estimator.best_score_ < self.min_score:
                continue

            n_inliers = len(this_subset)
            if n_inliers >= self.stop_n_inliers:
                break

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(this_X, this_y)
        self.inlier_inds_ = this_subset

        return self
