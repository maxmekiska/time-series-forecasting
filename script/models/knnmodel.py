from sklearn.neighbors import KNeighborsRegressor


KNNHYPARAM = {
"n_neighbors": [3, 5, 7, 8, 10, 12],
"weights":['uniform', 'distance'],
"algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
"leaf_size": [30, 40, 50],
"p":[1, 2, 3, 4],
"metric":['minkowski']
}
