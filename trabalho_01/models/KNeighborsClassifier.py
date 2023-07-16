import pandas as pd
import numpy as np


class KNeighborsClassifier():
    
    __class_names = []
    
    def __init__(self, n_neighbors=5):
        
        self.n_neighbors = n_neighbors
        
    def __euclidian_dist(self, point1, point2):
    
        point1 = np.array(point1)
        point2 = np.array(point2)

        sum_sq = np.sum(np.square(point1 - point2))

        return np.sqrt(sum_sq)
    
    def fit(self, X, y):
        
        X['target'] = y
        self.__class_names = list(X['target'].unique())
        self.df_train = X
        
    def __get_neighbors(self, X):
        
        df_train = self.df_train.copy()
        
        point1 = X.iloc[0,:].tolist()
        list_dist = []
        
        for index, _ in df_train.iterrows():
            
            point2 =  df_train.drop(['target'], axis=1).iloc[index, :].tolist()
            list_dist.append( self.__euclidian_dist(point1, point2) )
        
        df_train['distance_new_point'] = list_dist
        
        return (
            
            df_train[['target', 'distance_new_point']].sort_values(by=['distance_new_point'], ascending=True)
            
        )[:self.n_neighbors]
        
    def __predict_one(self, X):
        
        df_neighbors = self.__get_neighbors(X).reset_index(drop=True)
        
        class_predict = (
            
            df_neighbors.groupby('target', as_index=False).count()
                        .sort_values(by=['distance_new_point'], ascending=False)
        
        ).iloc[0,0]
        
        return class_predict
        
    def predict(self, X):

        list_class = []
        
        for index, _ in X.iterrows():
            
            class_predict = self.__predict_one(X.iloc[[index]])
            list_class.append(class_predict)
            
        return list_class
            
        
    def __predict_proba_one(self, X):
        
        df_neighbors = self.__get_neighbors(X).reset_index(drop=True)
        
        df_class = df_neighbors['target'].value_counts(ascending=False).to_frame()
        df_class.columns = ['frequencia']
        df_class['target'] = df_class.index.values
        
        df_class['proba'] = df_class['frequencia'] / (df_class['frequencia'].sum())
        
        list_return = []
    
        for class_name in self.__class_names:
            
            if class_name in list(df_class['target'].unique()):
                list_return.append( df_class[df_class['target'] == class_name].iloc[0,2])
            
            else:
                list_return.append(0.0)
        
        return list_return
    
    def predict_proba(self, X):
        
        list_class = []
        
        for index, _ in X.iterrows():
            
            class_proba = self.__predict_proba_one(X.iloc[[index]])
            list_class.append(class_proba)
            
        return list_class
    
    