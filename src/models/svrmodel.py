from sklearn.svm import SVR


SVRHYPARAM = {
"pipe__base_estimator__kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
"pipe__base_estimator__gamma": ['scale', 'auto'],
"pipe__base_estimator__tol": [1e-2, 1e-3, 1e-4],
"pipe__base_estimator__C": [0.5, 1, 2, 3],
"pipe__base_estimator__epsilon": [0.05, 0.1, 0.3],
"pipe__base_estimator__shrinking": [True, False]
}
