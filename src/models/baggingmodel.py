from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

BAGGINGHYPARAM = {
"base_estimator":[DecisionTreeRegressor(), KNeighborsRegressor(), SVR()],
"n_estimators": [5, 10, 15],
"max_samples":[1, 2, 3, 4],
"max_features": [1, 2, 3, 4],
"bootstrap": [True, False],
"bootstrap_features":[False, True]
}
