
import numpy as np
import numpy.testing as npt
from sklearn.ransac import RANSAC
import IPython
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM

def _scorer(m, X, y=None):
    return m.score(X).sum()

def test_ransacgmm():
    mus = np.array([(0, 0), (1, 1.41), (-10, -10)])
    stds = np.array([(0.2, 0.4), (0.5, 0.1), (1, 1)])
    weights = [50, 50, 1]

    x = []
    y = []
    for k in xrange(len(mus)):
        x.extend(np.random.normal(mus[k][0], stds[k][0], size=weights[k]))
        y.extend(np.random.normal(mus[k][1], stds[k][1], size=weights[k]))
    x = np.array(x)
    y = np.array(y)
    ind = np.random.permutation(np.arange(len(x)))
    x = x[ind]
    y = y[ind]

    X = np.vstack([x, y]).T

    gmm = GMM(n_components=2, n_init=5, min_covar=0.1)
    gscv = GridSearchCV(gmm, param_grid={'n_components' : np.arange(1, 10)}, 
                        cv=5, scoring=_scorer)

    ransac = RANSAC(gscv, min_samples=13)
    ransac.fit(X)

    # sometimes it gets overfit since there's not a lot of data,
    # but we just want to be sure, there's no mean around (-10, -10), 
    # and that the means are at least close to one of the actual means
    for mu in ransac.estimator_.best_estimator_.means_:
        dists = np.sqrt(np.square(mus - mu.reshape((1, -1))).sum(1))
        assert dists[2] > 10
        assert (dists[0] < 0.5) or (dists[1] < 0.5)

if __name__ == '__main__':
    test_ransacgmm()
