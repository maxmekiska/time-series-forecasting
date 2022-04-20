from sklearn.svm import LinearSVR

LSVMHYPARAM = {
"pipe__base_estimator__loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],
"pipe__base_estimator__epsilon":[0.0, 0.1, 0.3, 0.6],
"pipe__base_estimator__C": [1.0, 1.5, 2, 3],
"pipe__base_estimator__tol":[1e-4, 1e-5]
}
