from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


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

    
        
    def _sliding_window(self, _list, look_back, look_front):
        ''' Method divide input data into a sequential training dataset.
            
            Parameters:
                _list (list): Data to be devided into training dataset.
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
    
    def get_sequences(self):
        return self.X, self.y

    def _normalize_data(self, data):
        scaler = StandardScaler().fit(data)
        #scaled_data = scaler.transform(data)
        return scaler #scaled_data


    def forecast(self, data):
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
