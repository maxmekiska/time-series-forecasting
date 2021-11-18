from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



def grids(model: object, parameters: dict, X_train: list, y_train: list):
    grid = GridSearchCV(model, parameters, verbose=1)
    grid.fit(X_train, y_train)
    optimal_parameters = grid.best_params_
    print('Grid Search ' + model)
    print('Best score R2:', grid.best_score_)
    return optimal_parameters

def grids_halv(model: object, parameters: dict, X_train: list, y_train: list):
    grid = HalvingGridSearchCV(model, parameters, verbose=1)
    grid.fit(X_train, y_train)
    optimal_parameters = grid.best_params_
    print('Halving Grid Search ' + model)
    print('Best score R2:', grid.best_score_)

def grids_random(model: object, parameters: dict, X_train: list, y_train: list):
    grid = RandomizedSearchCV(model, parameters, verbose=1)
    grid.fit(X_train, y_train)
    optimal_parameters = grid.best_params_
    print('Randomized Grid Search ' + model)
    print('Best score R2:', grid.best_score_)
