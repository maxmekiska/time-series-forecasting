from sklearn.ensemble import AdaBoostRegressor

ADABOOSTHYPARAM = {
"pipe__base_estimator__n_estimators": [50, 70, 100, 150],
"pipe__base_estimator__learning_rate": [0.5, 1.0, 1.5, 2.0],
"pipe__base_estimator__loss": ['linear', 'square', 'exponential']
}
