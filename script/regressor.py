from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import mean, median, var
from pandas import DataFrame

from models.knnmodel import *
from models.dtreemodel import *
from models.randomforestmodel import *
from models.lsvmmodel import *
from models.adaboostmodel import *
from models.bayesianridgemodel import *
from models.gaussianprocessmodel import *
from models.baggingmodel import *
from models.svrmodel import *
from utils.gridsearch import *

class Regressor:
    ''' Class that wrapps around scikit learn time series regressors. Supports training, performance assessment and prediction functionality.

        Methods
        -------
            _sliding_window(self, _list: list, look_back: int, look_front: int):
                Private method divide input data into a sequential training dataset. 
            _normalize_data(self, data: list) -> object:
                Private method to normalize data. 
            get_X(self):
                Getter method to return X data set.  
            get_y(self):
                Getter method to return y data set.  
            get_Xtrain(self):
                Getter method to return X train data set.  
            get_ytrain(self):
                Getter method to return y train data set.  
            get_Xtest(self):
                Getter method to return X test data set.  
            get_ytest(self):
                Getter method to return y test data set.  
            performance(self, metric: str) -> None:
                Method to benchmark algorithm perfromance. Trainings data-set 80%, testing data-set 20%. 
            forecast(self, data: list) -> (list, list, list):
                Method to apply regression models onto target data.
    '''
    
    def __init__(self, time_series: list, look_back: int, look_future: int, scale: bool = False) -> object:
        '''
            Parameters:
                time_series (list): Input data for model.
                look_back (int): Steps predictor will look backward.
                look_future (int): Steps predictor will look forward.
                scale (bool): Scale input time series.
        '''
        self.X, self.y = self._sliding_window(time_series, look_back, look_future)

        self.scaler = self._normalize_data(self.X)

        self.scale = scale
        
        if self.scale == True:
            self.X = self.scaler.transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False, stratify=None)
        
        self.models = {"K-Neighbors Regressor": KNeighborsRegressor(), "DecisionTree Regressor": DecisionTreeRegressor(), "Random Forest Regressor": RandomForestRegressor(), "LinearSVR Regressor": RegressorChain(LinearSVR()), "Bayesian Ridge Regressor": RegressorChain(BayesianRidge()), "Ada Boost Regressor": RegressorChain(AdaBoostRegressor()), "Gaussian Process Regressor": GaussianProcessRegressor(), "Bagging Regressor": BaggingRegressor(), "SV Regressor": RegressorChain(SVR())}

        self.models_un = {"K-Neighbors Regressor": KNeighborsRegressor, "DecisionTree Regressor": DecisionTreeRegressor, "Random Forest Regressor": RandomForestRegressor, "LinearSVR Regressor": LinearSVR, "Bayesian Ridge Regressor": BayesianRidge, "Ada Boost Regressor": AdaBoostRegressor, "Gaussian Process Regressor": GaussianProcessRegressor, "Bagging Regressor": BaggingRegressor, "SV Regressor": SVR}

        self.models_optimized = {}

        self.hyperparameters = {"K-Neighbors Regressor": KNNHYPARAM, "DecisionTree Regressor": DTREEHYPARAM, "Random Forest Regressor": RFORESTHYPARAM, "LinearSVR Regressor": LSVMHYPARAM, "Bayesian Ridge Regressor": BAYRIDGEHYPARAM, "Ada Boost Regressor": ADABOOSTHYPARAM, "Gaussian Process Regressor": GAUSSHYPARAM, "Bagging Regressor": BAGGINGHYPARAM, "SV Regressor": SVRHYPARAM}

            
    def _sliding_window(self, _list: list, look_back: int, look_front: int):
        ''' Private method divide input data into a sequential training dataset.
            
            Parameters:
                _list (list): Data to be divided into training dataset.
                look_back (int): Days to look back.
                look_front (int): Days to predict into the future.
            Returns:
                back ([list]): List containing look back data lists. (X)
                front ([list]): List containing look into future data lists. (y)
        '''
        back = []
        front = []
        length_list = len(_list)
        for i in range(length_list):
            if i > length_list - look_back - look_front:
                break
            back.append(_list[i:i+look_back])
            front.append(_list[i+look_back : i+look_back+look_front])
        return back, front


    def _normalize_data(self, data: list) -> object:
        ''' Private method to normalize data.

            Parameters:
                data (list): Target data to be scaled.
            Returns:
                scaler (object): Sci-kit scaler object.
        '''
        scaler = StandardScaler().fit(data)
        return scaler
    
    def get_X(self):
        ''' Getter method to return X data set.'''
        return self.X

    def get_y(self):
        ''' Getter method to return y data set.'''
        return self.y
    
    def get_Xtrain(self):
        ''' Getter method to return X train data set.'''
        return self.X_train

    def get_ytrain(self):
        ''' Getter method to return y train data set.'''
        return self.y_train

    def get_Xtest(self):
        ''' Getter method to return X test data set.'''
        return self.X_test

    def get_ytest(self):
        ''' Getter method to return y test data set.'''
        return self.y_test

    def _clear_dict(self, dictionary: dict) -> dict:
        new_dict = {}
        for i, j in dictionary.items():
            new_dict[i[22:]] = j
        return new_dict

    def optimizer(self) -> None:
        models = self.models
        models_un = self.models_un
        params = self.hyperparameters
        for i, j in models.items():
            model = j
            parameter = params.get(i)
            model_un = self.models_un.get(i)
            if type(j) != RegressorChain: 
                optimized_params = grids_random(model, parameter, self.X_train, self.y_train) 
                self.models_optimized[i] = model_un(**optimized_params)
            else:
                _pipe = (Pipeline([('pipe', RegressorChain(model_un()))]))
                optimal_params = grids_random(_pipe, parameter, self.X_train, self.y_train) 
                optimal_params = self._clear_dict(optimal_params)
                self.models_optimized[i] = RegressorChain(model_un(**optimal_params))

    def optimize_ind(self, model: str) -> None:
        model_target = self.models.get(model)
        parameters = self.hyperparameters.get(model)
        model_un = self.models_un.get(model)
        if type(model_target) != RegressorChain:
            optimal_params = grids_random(model_target, parameters, self.X_train, self.y_train)
            self.models_optimized[model] = model_un(**optimal_params)
        else:
            _pipe = (Pipeline([('pipe', model_target)]))
            optimal_params = grids_random(_pipe, parameters, self.X_train, self.y_train) 
            optimal_params = self._clear_dict(optimal_params)
            self.models_optimized[model] = RegressorChain(model_un(**optimal_params)) 
        
    def performance(self, metric: str, optimized: bool = False) -> None:
        ''' Method to benchmark algorithm perfromance. Trainings data-set 80%, testing data-set 20%.

            Parameters:
                metric (str): What evaluation metric should be used, MSE or MAE.
        '''
        if metric == 'MSE':
            met = mean_squared_error
        elif metric == 'MAE':
            met = mean_absolute_error
        else:
            raise 'choose MSE or MAE'
       
        if optimized == False:
            models = self.models
        else:
            models = self.models_optimized

        results = {'Mean': [], 'Median': [], 'Variance': []}

        results = DataFrame(data=results)

        for i, j in models.items():
            performance_score_model = []

            model  = j
            model.fit(self.X_train, self.y_train)

            print(i)
            for k in tqdm(range(len(self.X_test))):
                data = [self.X_test[k]]
                yhat = model.predict(data)
                performance_score_model.append(met(yhat[0], self.y_test[k]))


            mean_value = mean(performance_score_model)
            median_value = median(performance_score_model) 
            variance_value = var(performance_score_model) 

            row = {'Mean': mean_value, 'Median': median_value, 'Variance': variance_value}
            row = DataFrame(data=row, index = [i])

            results = results.append(row)


            plt.plot(performance_score_model)
            plt.ylabel(metric)
            plt.title(i)
            plt.show()
            plt.clf()


        def minimum_value_in_column(column):    

            highlight = 'background-color: palegreen;'
            default = ''

            minimum_in_column = column.min()


            return [highlight if v == minimum_in_column else default for v in column]


        return results.style.apply(minimum_value_in_column, subset=['Mean', 'Median', 'Variance'], axis=0)

    def forecast_all(self, data: list, optimized: bool = False) -> list:
        ''' Method to apply regression models onto target data.

            Parameters:
                data (list): Target data for which prediction will be made.
            Returns:
                yhat_1, yhat_2, yhat3 [list, list, list]: Output predictions of all Regression models.
        '''
        if self.scale == True:
            data = self.scaler.transform([data])
        else:
            data = [data]

        if optimized == False:
            models = self.models
        else:
            models = self.models_optimized
        
        final_yhat = []
        for i, j in models.items():
            current_model = j
            current_model.fit(self.X, self.y)
            yhat = current_model.predict(data)
            final_yhat.append((i, yhat[0]))
        
        return final_yhat


    def forecast(self, data: list, model: str, optimized: bool = False) -> list:
        ''' Method to apply single chosen regression model onto target data.

            Parameters:
                data (list): Target data for which prediction will be made.
                model (str): What regressor model should be used.
            Returns:
                yhat_1, yhat_2, yhat3 (list, list, list): Output predictions of all Regression models.
        '''
        if self.scale == True:
            data = self.scaler.transform([data])
        else:
            data = [data]
        
        if optimized == False:
            models = self.models
        else:
            models = self.models_optimized

        model_used = models.get(model)

        model_used.fit(self.X, self.y)

        yhat = model_used.predict(data)
        
        return yhat
