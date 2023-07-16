import pandas as pd
import numpy as np
import math

class NormalMultivariate():

    def __init__(self, df, mean=None, cov=None):
        self.df = df
        self.dim = len(df.columns)

        if mean is None: self.mean = np.array([df[col].mean() for col in df.columns  if col != 'target'])
        else: self.mean = np.array(mean)

        if cov is None: self.cov = np.matrix(df.cov().values)
        else: self.cov = np.matrix(cov)

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

class GaussianMixtureModel(): 

    def __init__(self, dict_k, iter=2, threshold=0.01):
        self.threshold = threshold
        self.dict_k = dict_k # {class: n_component}
        self.means = {}
        self.means_past = {}
        self.iter = iter

    def __initialize(self, verbose):
      
      if verbose:
          print('*' * 40, 'INICIALIZATION', '*' * 40)
      self.coef_mixture = {}
      for j in sorted(self.list_class):
          self.dict_norm_class[j] = {}
          self.coef_mixture[j] = {}
          if verbose:
              print(f'Class {j}')
          for k, row in zip(range(self.dict_k[j]), self.dict_df_class[j][self.features].sample(n=self.dict_k[j], random_state=42).reset_index(drop=True).iterrows()):
              mean = row[1].tolist()
              self.dict_norm_class[j][k] = NormalMultivariate(self.dict_df_class[j][self.features], mean)  
              self.coef_mixture[j][k] = 1 / self.dict_k[j] # Iniciando Equiprovável

              if verbose:
                  print(f'     Component {k}')
                  print('          Coeficient Mixture:', self.coef_mixture[j][k])
                  print('          Gaussian:')
                  print('               Média da Gaussian:', mean)
                  print('               Covariancia:', self.dict_norm_class[j][k].cov.tolist())

    def __init_list_dict(self, class_):
        list_components = {}
        for k in range(self.dict_k[class_]):
            list_components[k] = [0] * len(self.dict_df_class[class_])
        return list_components

    def __expetation(self, verbose):
        if verbose:
            print('\n', '*' * 40, 'EXPETATION STEP', '*' * 40)
        self.dict_df_components = {} 
        for j in sorted(self.list_class):
            
            if verbose:
                print(f'Class {j}')
            list_components = self.__init_list_dict(j)
            list_component_choose = []
            list_verossimilhanca = []
            
            for idx, _ in self.dict_df_class[j].iterrows():

                x = self.dict_df_class[j][self.features].iloc[idx].tolist()
                list_posteriori = []
                total = 0
                for k in range(self.dict_k[j]):
                    posteriori = self.coef_mixture[j][k] * self.dict_norm_class[j][k].pdf(x)
                    list_posteriori.append(posteriori) # Posteriori
                    total += posteriori
                
                list_verossimilhanca.append(total)
                normalizator = sum(list_posteriori) # Normaliador
                component = 0
                max_reponsability = -np.inf
                for k in range(self.dict_k[j]):
                    # Compute responsability
                    pdf = list_posteriori[k]
                    responsability = pdf / normalizator
                    list_components[k][idx] = responsability
                    if max_reponsability < responsability:
                        max_reponsability = responsability
                        component = k
                list_component_choose.append(component)
            
            self.dict_df_components[j] = pd.DataFrame.from_dict(list_components) # Cada coluna é uma component
            self.dict_df_components[j]['component_choose'] = list_component_choose
            self.dict_df_components[j]['probability'] = list_verossimilhanca
            if verbose:
                display(self.dict_df_components[j].head(5))

    def __maximization(self, verbose):
        if verbose:
            print('\n', '*' * 40, 'MAXIMIZATION STEP', '*' * 40)
        for j in sorted(self.list_class):
            if verbose:
                print(f'Class {j}')
            self.means[j] = {}
            self.means_past[j] = {}
            df_class = self.dict_df_class[j][self.features].copy()
            df_class = df_class.join(self.dict_df_components[j])
            for k in range(self.dict_k[j]):
                if verbose:
                  print(f'Component {k}')
                responsability = df_class[k]
                row = df_class[self.features].values

                sum_mean = np.array([0] * (len(self.features)))
                for x, resp in zip(row, responsability):
                    sum_mean = sum_mean + (x * resp)

                # Updating Mean
                new_mean = sum_mean / sum(responsability)
                
                # Updating Coef Mixture
                self.coef_mixture[j][k] = sum(responsability) / len(responsability)

                # Updating covariance
                list_diff = []
                for x in row:
                    list_diff.append(np.array(x) - np.array(new_mean))
                
                diff = (row - np.array([new_mean] * len(row))).T
                weighted_sum = np.dot(np.array(responsability) * diff, diff.T)
                new_cov = weighted_sum / sum(responsability)

                self.means_past[j][k] = self.dict_norm_class[j][k].mean

                # Updating Gaussian
                self.dict_norm_class[j][k] = NormalMultivariate(self.dict_df_class[j][self.features], new_mean, new_cov)

                self.means[j][k] = new_mean
                if verbose:
                    print('New Mean', new_mean)
                    print('New cov', new_cov)
                    print()


    def fit(self, X_train, y_train, verbose=False):
        df_train = X_train.copy()
        df_train['target'] = y_train
        self.list_class = df_train['target'].unique()
        self.features = [col for col in df_train.columns if col != 'target']
    
        self.dict_priori = {}
        self.dict_df_class = {}
        self.dict_norm_class = {}
    
        for j in sorted(self.list_class):
            self.dict_priori[j] = (len(df_train.loc[df_train['target'] == j]) / len(df_train))
            self.dict_df_class[j] = df_train[df_train['target'] == j].reset_index(drop=True).copy()
             
        self.__initialize(verbose)
        for i in range(self.iter):
            self.__expetation(verbose)
            self.__maximization(verbose)
            check = True
            for j in sorted(self.list_class):
                for k in range(self.dict_k[j]):
                    if not all(i <= self.threshold for i in abs(np.array(self.means[j][k]) - np.array(self.means_past[j][k]))):
                        check = False
            if check:
                break

    def predict(self, df):

        list_predict = []
        for idx, row in df.iterrows():
    
            x = df[self.features].iloc[idx].tolist()
    
            pdf_priori = []
            dict_pdf = {}
            for j in sorted(self.list_class):
                verossimilhanca = 0
                for k in range(self.dict_k[j]):
                    verossimilhanca += (self.dict_norm_class[j][k].pdf(x) * self.coef_mixture[j][k])
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