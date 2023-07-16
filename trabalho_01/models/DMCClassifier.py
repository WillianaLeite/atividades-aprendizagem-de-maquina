import pandas as pd
import numpy as np


class DMCClassifier():
    
    __class_names = []
    
    def __euclidian_dist(self, point1, point2):
    
        point1 = np.array(point1)
        point2 = np.array(point2)

        sum_sq = np.sum(np.square(point1 - point2))

        return np.sqrt(sum_sq)
    
    def fit(self, X, y):
        
        X['target'] = y
        X = X.groupby('target', as_index=False).mean() # Calculando os Centróides por classe
        self.__class_names = list(X['target'].unique())
        self.df_train = X
        
    def __get_neighbors(self, X):
        
        df_train = self.df_train.copy()
        
        point1 = X.iloc[0,:].tolist()
        list_dist = []
        
        for index, row in df_train.iterrows():
            
            point2 =  df_train.drop(['target'], axis=1).iloc[index, :].tolist()
            list_dist.append( self.__euclidian_dist(point1, point2) )
        
        df_train['distance_new_point'] = list_dist
        
        return df_train[['target', 'distance_new_point']]
        
    def __predict_one(self, X):
        
        df_train = self.__get_neighbors(X)
        class_predict = (
            
            df_train[['target', 'distance_new_point']].sort_values(by=['distance_new_point'], ascending=True)
            
        ).iloc[0,0]
        
        return class_predict
        
    def predict(self, X):

        list_class = []
        
        for index, _ in X.iterrows():
            
            class_predict = self.__predict_one(X.iloc[[index]])
            list_class.append(class_predict)
        
        return list_class          
    