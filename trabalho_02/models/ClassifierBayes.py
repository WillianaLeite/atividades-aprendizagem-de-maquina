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


class NormalUnivariate():

    def __init__(self, array):
        self.array = array
        self.mean = np.mean(array)
        self.std = np.std(array)
        self.var = self.std ** 2
        self.peak_distribution = 1 / (np.sqrt(2 * np.pi * self.var))

    def pdf(self, x):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            return [self.peak_distribution * (np.exp(-(i - self.mean) ** 2 / (2 * self.var))) for i in x]

        return self.peak_distribution * (np.exp(-(x - self.mean) ** 2 / (2 * self.var)))

    def plot_pdf(self, x=None):
        if x is None:
            xmin, xmax = min(self.array), max(self.array)
            x = np.linspace(xmin, xmax, 100)

        probabilities = self.pdf(x)
        plt.hist(self.array, density=True, color='g', alpha=0.6, bins=18)
        plt.plot(x, probabilities, 'k', linewidth=2)
        plt.title(f'MÃ©dia = {self.mean},  std = {self.std}')
        plt.show()
        
class NormalMultivariate():

    def __init__(self, df):
        self.df = df
        self.dim = len(df.columns)
        self.mean = np.array([df[col].mean() for col in df.columns  if col != 'target'])
        self.cov = np.matrix(df.cov().values)
        self.det = np.linalg.det(self.cov)
        self.peak_distribution = 1 / (((2 * np.pi) ** (self.dim / 2)) * 
                                      (self.det ** 0.5))
        self.pseudo_peak_distribution = None
    
    def __compute_pseudo_metrics(self):
        self.pseudo_inv = np.linalg.pinv(self.cov)
        eig_values = np.linalg.eig(self.cov)
        self.pseudo_determinent = np.product([i for i in eig_values[0] if i > 1e-12])
        self.pseudo_peak_distribution = 1 / (((2 * np.pi) ** (self.dim / 2)) * 
                                             (self.pseudo_determinent ** 0.5))

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
        try:
            inv = self.cov.I
            return self.peak_distribution * math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        except np.linalg.LinAlgError as err:
            if self.pseudo_peak_distribution is None:
                self.__compute_pseudo_metrics()
            return self.pseudo_peak_distribution * math.pow(math.e, -0.5 * (x_mu * self.pseudo_inv * x_mu.T))
             
        


    def plot_pdf(self, col_x1, col_x2):
        
        if self.dim > 2:
            raise NameError('Viewing can only be done with 2 attributes')
        x = np.linspace(self.df[col_x1].min(), self.df[col_x1].max(), 100)
        y = np.linspace(self.df[col_x2].min(), self.df[col_x2].max(), 100)
        df = pd.DataFrame(list(zip(x,y)), columns=[col_x1, col_x2])
        z = self.pdf(df)
        df['prob'] = z

        threedee = plt.figure(figsize=(15, 12)).gca(projection='3d')
        threedee.plot(df[col_x1], df[col_x2], df['prob'])
        threedee.set_xlabel(col_x1)
        threedee.set_ylabel(col_x2)
        threedee.set_zlabel('prob')

        plt.title(f'MÃ©dia = {self.mean},  cov = {self.cov}')
        plt.show()

class ClassifierBayes():
    
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
            self.dict_df_class[j] = df_train[df_train['target'] == j].copy()
            self.dict_norm_class[j] = NormalMultivariate(self.dict_df_class[j][self.features])   

    
    def predict(self, df):

        list_predict = []
        for idx, row in df.iterrows():
    
            x = df[self.features].iloc[idx].tolist()
    
            pdf_priori = []
            dict_pdf = {}
            for j in sorted(self.list_class):
                verossimilhanca = self.dict_norm_class[j].pdf(x)
                dict_pdf[j] = verossimilhanca
                pdf_priori.append(
                    dict_pdf[j] * self.dict_priori[j]
                )
    
            normalize = sum(pdf_priori)
            prob_class = {}
            prob_max = -1
            class_predict = -1
            for j in sorted(self.list_class): 
                prob_class[j] =  (dict_pdf[j] * self.dict_priori[j]) / normalize
                if prob_class[j] > prob_max:
                    prob_max = prob_class[j]
                    class_predict = j
    
            list_predict.append(class_predict)
        
        return list_predict