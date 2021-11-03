from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import mean, median, var


class Regressor:
    
    def __init__(self, time_series, look_back, look_future, scale=False) -> object:
        '''
            Parameters:
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
                data (DataFrame): Input data for model training.
        '''
        self.X, self.y = self._sliding_window(time_series, look_back, look_future)

        self.scaler = self._normalize_data(self.X)

        self.scale = scale
        
        if self.scale == True:
            self.X = self.scaler.transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False, stratify=None)
        

            
    def _sliding_window(self, _list, look_back, look_front):
        ''' Method divide input data into a sequential training dataset.
            
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

    def _normalize_data(self, data: list) -> object:
        ''' Private method to normalize data.

            Parameters:
                data (list): Target data to be scaled.
            Returns:
                scaler (object): Sci-kit scaler object.
        '''
        scaler = StandardScaler().fit(data)
        return scaler

    def performance(self, metric: str) -> None:
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

        performance_score_model_1 = []
        performance_score_model_2 = []
        performance_score_model_3 = []

        model_1 = KNeighborsRegressor()
        model_2 = DecisionTreeRegressor()
        model_3 = RandomForestRegressor()
        model_1.fit(self.X_train, self.y_train)
        model_2.fit(self.X_train, self.y_train)
        model_3.fit(self.X_train, self.y_train)
        
        print('K-Neighbors Regressor Model')
        for i in tqdm(range(len(self.X_test))):
            data = [self.X_test[i]]
            yhat_1 = model_1.predict(data)
            performance_score_model_1.append(met(yhat_1[0], self.y_test[i]))

        plt.plot(performance_score_model_1)
        plt.ylabel(metric)
        plt.title('K-Neighbors Regressor Model Performance')
        plt.show()
        plt.clf()

        print('Decision-Tree Regressor Model')
        for i in tqdm(range(len(self.X_test))):
            data = [self.X_test[i]]
            yhat_1 = model_2.predict(data)
            performance_score_model_2.append(met(yhat_1[0], self.y_test[i]))

        plt.plot(performance_score_model_2)
        plt.ylabel(metric)
        plt.title('Decision-Tree Regressor Model Performance')
        plt.show()
        plt.clf()

        print('Random-Forest Regressor Model')
        for i in tqdm(range(len(self.X_test))):
            data = [self.X_test[i]]
            yhat_1 = model_3.predict(data)
            performance_score_model_3.append(met(yhat_1[0], self.y_test[i]))

        plt.plot(performance_score_model_3)
        plt.ylabel(metric)
        plt.title('Random-Forest Regressor Model Performance')
        plt.show()
        plt.clf()
        
        mean1 = mean(performance_score_model_1)
        mean2 = mean(performance_score_model_2)
        mean3 = mean(performance_score_model_3)

        
        median1 = median(performance_score_model_1)
        median2 = median(performance_score_model_2)
        median3 = median(performance_score_model_3)

        var1 = var(performance_score_model_1)
        var2 = var(performance_score_model_2)
        var3 = var(performance_score_model_3)

        print(f'K-Neighbors Regressor {metric} Mean: {mean1} Median: {median1}, Variance: {var1}')
        print(f'Decision-Tree Regressor {metric} Mean: {mean2} Median: {median2}, Variance: {var2}')
        print(f'Random-Forest Regressor {metric} Mean: {mean3} Median: {median3}, Variance: {var3}')

    def forecast(self, data: list) -> (list, list, list):
        ''' Method to apply regression models onto target data.

            Parameters:
                data (list): Target data for which prediction will be made.
            Returns:
                yhat_1, yhat_2, yhat3 (list, list, list): Output predictions of all Regression models.
        '''
        if self.scale == True:
            data = self.scaler.transform([data])
        else:
            data = [data]

        model_1 = KNeighborsRegressor()
        model_2 = DecisionTreeRegressor()
        model_3 = RandomForestRegressor()
        model_1.fit(self.X, self.y)
        model_2.fit(self.X, self.y)
        model_3.fit(self.X, self.y)
        
        yhat_1 = model_1.predict(data)
        yhat_2 = model_2.predict(data)
        yhat_3 = model_3.predict(data)
        
        return yhat_1, yhat_2, yhat_3
