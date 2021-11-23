from sklearn.gaussian_process import GaussianProcessRegressor

GAUSSHYPARAM = {
"alpha": [1e-9, 1e-10, 1e-11],
"n_restarts_optimizer": [0, 1, 2],
"normalize_y": [False, True]
}
