import pandas as pd
import numpy as np


class ClassifierBayesDiscriminant():
    
    def __init__(self, discrimant='quadratico', type_cov='pool'):
        self.discrimant = discrimant
        if self.discrimant == 'linear':
            self.type_cov = type_cov

    def fit(self, X_train, y_train, verbose=False):
        df_train = X_train.copy()
        df_train['target'] = y_train
        self.list_class = df_train['target'].unique()
        self.features = [col for col in df_train.columns if col != 'target']
    
        self.dict_priori = {}
        self.dict_df_class = {}
        self.dict_mean = {}
        self.dict_cov = {}
        self.dict_inv_cov = {}
        self.dict_det = {}
        self.is_singular = False
        
        for j in sorted(self.list_class):
            self.dict_priori[j] = (len(df_train.loc[df_train['target'] == j]) / len(df_train))
            self.dict_df_class[j] = df_train[df_train['target'] == j].copy()
            self.dict_mean[j] = np.matrix([self.dict_df_class[j][col].mean() for col in self.dict_df_class[j].columns if col != 'target'])
            if self.discrimant == 'quadratico':
                mtx_cov = np.matrix(self.dict_df_class[j].drop(['target'], axis=1).cov().values)
                self.dict_cov[j] = mtx_cov
                self.dict_inv_cov[j] = self.__inv_mtx(self.dict_cov[j])
                self.dict_det[j] = self.__det_mtx(self.dict_cov[j])
        
        if (self.discrimant == 'linear'):
            cov_linear = None
            if(self.type_cov == 'pool'):
                cov_linear = sum([self.dict_priori[j] * np.matrix(self.dict_df_class[j].drop(['target'], axis=1).cov().values) for j in sorted(self.list_class)])
            elif (self.discrimant == 'linear') & (self.type_cov == 'all_data'):
                cov_linear = np.matrix(df_train.drop(['target'], axis=1).cov().values)
            for j in sorted(self.list_class):
                self.dict_cov[j] = cov_linear
                self.dict_inv_cov[j] = self.__inv_mtx(self.dict_cov[j])
                self.dict_det[j] = self.__det_mtx(self.dict_cov[j])
        if verbose:
            print('media: ', self.dict_mean[j]) 
            print('inversa mtx covariancia: ', self.dict_inv_cov) 
            print('determinante: ',self.dict_det) 

    def __inv_mtx(self, cov):
        try:
            return cov.I
        except np.linalg.LinAlgError as err:
            self.is_singular = True
            return np.linalg.pinv(cov)
    
    def __det_mtx(self, cov):
        if self.is_singular:
            eig_values = np.linalg.eig(cov)
            return np.product([i for i in eig_values[0] if i > 1e-12])
        else:
            return np.linalg.det(cov)
    
    def predict(self, df, verbose=False):

        list_predict = []
        for idx, row in df.iterrows():
    
            x = np.matrix([df[self.features].iloc[idx].tolist()])
            dict_discriminating = {}
            max_disc = -np.inf
            class_predict = -np.inf
            for j in sorted(self.list_class):
                if verbose:
                    print('x.T', x.T)
                    print('inv_cov', self.dict_inv_cov[j])
                    print('x', x)
                    print('1º', -0.5 * float(x * self.dict_inv_cov[j] * x.T))
                    print('2º', 0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * x.T))
                    print('3º', 0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * self.dict_mean[j].T))
                    print('4º', 0.5 * float(x * self.dict_inv_cov[j] * self.dict_mean[j].T))
                    print('5º', np.log(self.dict_priori[j]))
                    print('6º', -0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.dict_det[j]))
                if self.discrimant == 'linear':
                    dict_discriminating[j] = (
                        (0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * x.T)) - # 1 º
                        (0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * self.dict_mean[j].T)) + # 2 º
                        (0.5 * float(x * self.dict_inv_cov[j] * self.dict_mean[j].T)) + # 3 º
                        np.log(self.dict_priori[j])
                    )
                elif self.discrimant == 'quadratico':
                    dict_discriminating[j] = (
                        (-0.5 * float(x * self.dict_inv_cov[j] * x.T)) + # 1º
                        (0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * x.T)) - # 2 º
                        (0.5 * float(self.dict_mean[j] * self.dict_inv_cov[j] * self.dict_mean[j].T)) + # 3 º
                        (0.5 * float(x * self.dict_inv_cov[j] * self.dict_mean[j].T)) + # 4 º
                        np.log(self.dict_priori[j]) + # 5º
                        ((-0.5 * (np.log(2 * np.pi)) - (0.5 * np.log(self.dict_det[j])))) #ci

                    )

                if dict_discriminating[j] > max_disc:
                    max_disc = dict_discriminating[j]
                    class_predict = j
            if verbose:
                print(f'Linha {idx} discriminantes: ')
                print(dict_discriminating)
            list_predict.append(class_predict)
        
        return list_predict