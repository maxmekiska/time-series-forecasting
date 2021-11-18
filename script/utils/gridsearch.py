from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV


def grids(model: object, parameters: dict, X_train: list, y_train: list):
    grid = GridSearchCV(model, parameters, verbose=1)
    grid.fit(X_train, y_train)
    optimal_parameters = grid.best_params_
    print('Best score:', grid.best_score_)
    return optimal_parameters
