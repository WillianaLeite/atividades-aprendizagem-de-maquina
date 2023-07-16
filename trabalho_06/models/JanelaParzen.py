import pandas as pd
import numpy as np
import seaborn as sns
import math
import scipy
from mpl_toolkits import mplot3d
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

import matplotlib.pyplot as plt


class NormalMultivariate():

    def __init__(self, df, mean=None, cov=None):
        self.df = df
        self.dim = len(df.columns)

        if mean is None: self.mean = np.array([df[col].mean() for col in df.columns  if col != 'target'])
        else: self.mean = np.array(mean)

        if cov is None: self.cov = np.matrix(df.cov().values)
        else: self.cov = cov

        self.det = np.linalg.det(self.cov)
        self.peak_distribution = 1 / (((2 * np.pi) ** (self.dim / 2)) * 
                                      (self.det ** 0.5))

    def pdf(self, x):
        if isinstance(x, pd.core.frame.DataFrame) or isinstance(x[0], list) or isinstance(x[0], np.ndarray) or isinstance(x[0], np.matrix):
            if isinstance(x, pd.core.frame.DataFrame):
                x = x.values
            list_return = []
            for i in x:
                list_return.append(self.__pdf_single_list(i))
            return list_return

        else:
            return self.__pdf_single_list(x)

    def __pdf_single_list(self, x):

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x_mu = np.matrix(x - self.mean)
        inv = self.cov.I
        return self.peak_distribution * math.pow(math.e, -((x_mu * x_mu.T) / (2 * self.cov)))
  
class JanelaParzen():

    def __init__(self, len_window):
        self.k = len_window

    def fit(self, X_train, y_train):
        df_train = X_train.copy()
        df_train['target'] = y_train
        self.list_class = df_train['target'].unique()
        self.features = [col for col in df_train.columns if col != 'target']
    
        self.dict_priori = {}
        self.dict_df_class = {}
        self.dict_norm_class = {}
    
        for j in sorted(self.list_class):
            self.dict_priori[j] = (len(df_train.loc[df_train['target'] == j]) / len(df_train))
            self.dict_df_class[j] = df_train[df_train['target'] == j].copy().reset_index(drop=True)
            self.dict_norm_class[j] = {}
            for idx, _ in self.dict_df_class[j].iterrows():
                x = self.dict_df_class[j][self.features].iloc[idx].tolist()
                self.dict_norm_class[j][idx] = NormalMultivariate(self.dict_df_class[j][self.features], mean=x, cov=np.matrix([[self.k **2]]).I)



    def prob_x(self, xi):
        dict_prob = {}
        for j in sorted(self.list_class):
            list_gaussian = []
            for idx, _ in self.dict_df_class[j].iterrows():
                list_gaussian.append(self.dict_norm_class[j][idx].pdf(xi))
            dict_prob[j] = sum(list_gaussian) / len(self.dict_df_class[j])
            
        return dict_prob

    
    def predict(self, df):

        list_predict = []
        for idx, row in df.iterrows():
    
            x = df[self.features].iloc[idx].tolist()
    
            pdf_priori = []
            dict_posteriori =self.prob_x(x)
            for j in sorted(self.list_class):
                pdf_priori.append(
                    dict_posteriori[j] * self.dict_priori[j]
                )
    
            normalize = sum(pdf_priori)
            prob_class = {}
            prob_max = -1
            class_predict = -1
            for j in sorted(self.list_class): 
                prob_class[j] =  (dict_posteriori[j] * self.dict_priori[j]) / normalize
                if prob_class[j] > prob_max:
                    prob_max = prob_class[j]
                    class_predict = j
    
            list_predict.append(class_predict)
        
        return list_predict