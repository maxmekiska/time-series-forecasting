from sklearn.tree import DecisionTreeRegressor

DTREEHYPARAM = {
"criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
"splitter":['best', 'random'],
"max_features": ['auto', 'sqrt', 'log2']
}
