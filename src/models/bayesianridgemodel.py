from sklearn.linear_model import BayesianRidge

BAYRIDGEHYPARAM = {
"pipe__base_estimator__n_iter": [300, 350, 400, 450],
"pipe__base_estimator__tol":[1e-3, 1e-4],
"pipe__base_estimator__alpha_1": [1e-6, 1e-5, 1e-7],
"pipe__base_estimator__alpha_2": [1e-6, 1e-5, 1e-7],
"pipe__base_estimator__lambda_1": [1e-6, 1e-5, 1e-7],
"pipe__base_estimator__lambda_2": [1e-6, 1e-5, 1e-7]
}
