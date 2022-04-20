from sklearn.linear_model import TweedieRegressor

LTWEEDIEHYPARAM = {
"pipe__base_estimator__power": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
"pipe__base_estimator__alpha":[0, 0.5, 1, 2, 4],
"pipe__base_estimator__link":['auto', 'identity', 'log'],
"pipe__base_estimator__max_iter":[50, 100, 150, 300],
"pipe__base_estimator__tol": [1e-3, 1e-4, 1e-5],
"pipe__base_estimator__warm_start":[False, True]
}
