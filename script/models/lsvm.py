from sklearn.svm import LinearSVR

LSVRYPARAM = {
"loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],
"epsilon":[0.0, 0.1, 0.3, 0.6],
"C": [1.0, 1.5, 2, 3],
"tol":[1e-4, 1e-5]
}
