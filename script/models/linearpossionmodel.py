from sklearn.linear_model import PossionRegressor

LPOSSIONHYPARAM = {
"pipe__base_estimator__alpha": [0, 0.5, 1, 2, 5, 10],
"pipe__base_estimator__max_iter":[50, 100, 150, 300],
"pipe__base_estimator__warm_start": [1e-3, 1e-4, 1e-5],
"pipe__base_estimator__tol":[False, True]
}
