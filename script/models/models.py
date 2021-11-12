# non working example from:

#https://towardsdatascience.com/20x-times-faster-grid-search-cross-validation-19ef01409b7c#:~:text=Grid%20Search%20cross%2Dvalidation%20is,by%20a%20grid%20of%20hyperparameters.&text=Grid%20Search%20CV%20tries%20all,having%20the%20best%20performance%20score. 


from sklearn.model_selection import GridSearchCV

param_grid = {
"max_depth": [3, 4, 7, 10, 25],
"gamma": [0.5, 1, 5, 10, 25],
"min_child_weight": [1, 3, 5, 10, 25],
"reg_lambda": [5, 10, 50, 100, 300],
"scale_pos_weight": [1, 3, 5, 10, 25]
}


# Grid Search CV implementation
xgb_cl = xgb.XGBClassifier(objective="binary:logistic")

halving_cv = HalvingGridSearchCV(xgb_cl, param_grid, scoring="roc_auc", n_jobs=-1, min_resources="exhaust", factor=3)

halving_cv.fit(X_train, y_train)

# Return set of parameters with the best performance
halving_cv.best_params_

# Return the performance metric score
halving_cv.best_score_
