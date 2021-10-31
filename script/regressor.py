from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, shuffle=False, stratify=None)
        

            
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
    
    def get_X(self):
        return self.X

    def get_y(self):
        return self.y
    
    def get_Xtrain(self):
        return self.X_train

    def get_ytrain(self):
        return self.y_train

    def get_Xtest(self):
        return self.X_test

    def get_ytest(self):
        return self.y_test

    def _normalize_data(self, data):
        scaler = StandardScaler().fit(data)
        return scaler

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
