from sklearn.svm import LinearSVR

LSVRYPARAM = {
"pipe__estimator__loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],
"pipe__estimator__epsilon":[0.0, 0.1, 0.3, 0.6],
"pipe__estimator__C": [1.0, 1.5, 2, 3],
"pipe__estimator__tol":[1e-4, 1e-5]
}
