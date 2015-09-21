
import numpy as np
from .base import BaseEstimator, clone
from .grid_search import GridSearchCV
import scipy.spatial
import warnings

def _get_point_scores(model, X, y=None):
    if y is None:
        scores = [model.score(X[k:k + 1]) for k in xrange(X.shape[0])]
    else:
        scores = [model.score(X[k:k + 1], y[k:k + 1]) for k in xrange(X.shape[0])]

    scores = np.log(np.array(scores))

    return scores


def _uniform_initializer(X, n_samples):
    return np.random.permutation(np.arange(X.shape[0]))[:n_samples]


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
    initializer : callable, optional
        The function to re-initialize the subset of points
        it should take two arguments: X (the matrix of points
        being fit) and n_samples (the number of samples to
        grab)
    """
    def __init__(self, model, min_samples=3, max_trials=10, 
                 inlier_thresh=np.log(0.05), 
                 point_scorer=_get_point_scores, 
                 initializer=_uniform_initializer):

        if not isinstance(model, GridSearchCV):
            raise ValueError("model must be a GridSearchCV instance")
        self.base_estimator = model
        self.min_samples = min_samples
        self.max_trials = max_trials
        self.inlier_thresh = inlier_thresh
        self.point_scorer = point_scorer
        self.initializer = initializer


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
        best_model = {'inds' : [], 'score' : -np.inf, 
                      'model' : None}
        for k in xrange(1):#self.max_trials):
            last_score = -np.inf
            this_estimator = clone(self.base_estimator)
            #this_estimator.best_score_ = -np.inf
            this_subset = self.initializer(X, self.min_samples)

            this_X = X[this_subset]
            if supervised:
                this_y = y[this_subset]
            else:
                this_y = None

            for i in xrange(self.max_trials):
                print 'Trial %d.%d (%d inliers / %d points | %2d%%) [%f]' % (k, i, len(this_subset), n_samples, len(this_subset) / float(n_samples) * 100, last_score)
                try:
                    this_estimator.fit(this_X, this_y)
                except Exception as err:
                    warnings.warn(err.message)
                    raise err
                    
                    # let's just assume this is our fault
                    # and try restarting the iteration
                    # (most common problem is fewer points
                    # than parameters in the model.)
                    break

                scores = self.point_scorer(this_estimator, X, y)

                new_subset = np.where((scores > self.inlier_thresh))[0]

                if last_score == -np.inf:
                    num_samples = self.min_samples
                else:
                    diff = this_estimator.best_score_ - last_score + 10
                    # subtract ten so that when the scores are equal we try to
                    # add some new data
                    num_samples = len(this_X) + \
                        0.2 * len(X) * (1.0 / (1.0 + np.exp(-7e-3 * diff)) - 0.5)
                    #if this_estimator.best_score_ >= last_score:
                    #    num_samples = len(this_X) + 0.1 * len(X)
                    #else:
                    #    num_samples = 0.8 * len(this_X)

                    num_samples = np.sum(scores > np.mean(scores))

                #new_subset = self.initializer(X, num_samples,
                #                              likes=scores)

                this_subset = new_subset
                this_X = X[this_subset]
                if supervised:
                    this_y = y[this_subset]
                else:
                    this_y = None

                #if (this_estimator.best_score_ - last_score) < (-0.2 * np.abs(last_score)):
                #    break
                if np.abs(this_estimator.best_score_ - last_score) < 0.005 * np.abs(last_score):
                    break


                if this_estimator.best_score_ > best_model['score']:
                    best_model['inds'] = this_subset
                    best_model['model'] = this_estimator
                    best_model['score'] = this_estimator.best_score_

                last_score = this_estimator.best_score_

        this_subset = best_model['inds']
        this_X = X[this_subset]
        if supervised:
            this_y = y[this_subset]
        else:
            this_y = None

        self.estimator_ = clone(best_model['model'])
        self.estimator_.fit(this_X, this_y)
        self.inlier_inds_ = this_subset

        return self
