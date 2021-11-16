from sklearn.ensemble import RandomForestRegressor

RFORESTHYPARAM = {
"n_estimators": [100, 130, 150, 170],
"criterion": ['squared_error', 'absolute_error', 'poisson'],
"max_features": ['auto', 'sqrt', 'log2']
}
