import pandas as pd
import numpy as np
import random
import statsmodels.api as sm

def target_one_vs_all(df, col_name_target='target'):
    
    unique_target = df[col_name_target].unique().tolist()
    
    dict_df = {}
    
    for target in unique_target:
        
        df_ = df.copy()
        
        df_[col_name_target] = df_[col_name_target].apply(lambda row_target: 1 if row_target == target else 0)
        
        dict_df[target] = df_
        
    return dict_df

def __split_by_class(df, col_target, size, black_list=None):
    
    lista = []
    
    for class_ in df[col_target].unique():
            
        n = round(size * len(df[df[col_target] == class_])) # Calculando a quantidade de registros de acordo com o input
        
        list_index = df[df[col_target] == class_].index.values.tolist() # Pegando a lista de index da classe
        
        if black_list is not None: # pra não retornar index que já foram usados
            
            list_index = [idx for idx in list_index if idx not in black_list] 
        
        idx_list = random.sample(list_index, n)
        
        lista = lista + idx_list
        
    return lista


def split_train_test(df, col_target, train_size=0.7, test_size=None, valid_size=None, stratify=None):
    
    if stratify is None:
    
        columns_train = [col for col in df.columns if col != col_target] 

        total = len(df)
        n_train = round(train_size * total)

        indexs = list(range(total))

        idx_train = random.sample(indexs, n_train)

        df_train = df.iloc[idx_train]

        if (test_size is None) and (valid_size is None):  
            n_test = round((1 - train_size) * total)

        if(test_size is not None):
            n_test = round(test_size * total)

        idx_test = random.sample([idx for idx in indexs if idx not in idx_train], n_test)
        df_test = df.iloc[idx_test]
        
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        
        if(valid_size is None):
            
            return df_train[columns_train], df_train[col_target], df_test[columns_train], df_test[col_target]


        if(valid_size is not None):

            n_valid = round(valid_size * total)

            idx_valid = random.sample([idx for idx in indexs if ((idx not in idx_train) and (idx not in idx_test))], n_valid)

            df_valid = df.iloc[idx_valid]
            
            df_valid.reset_index(drop=True, inplace=True)

            return (df_train[columns_train], df_train[col_target], 
                    df_test[columns_train], df_test[col_target],
                    df_valid[columns_train], df_valid[col_target])
        
    else:
        
        columns_train = [col for col in df.columns if col != col_target] 

        idx_train = __split_by_class(df, col_target, train_size)

        df_train = df.iloc[idx_train]

        if (test_size is None) and (valid_size is None):  
            idx_test = __split_by_class(df, col_target, (1 - train_size), black_list=idx_train)

        if(test_size is not None):
            idx_test = __split_by_class(df, col_target, test_size, black_list=idx_train)

        df_test = df.iloc[idx_test]
        
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        if(valid_size is None):

            return df_train[columns_train], df_train[col_target], df_test[columns_train], df_test[col_target]


        if(valid_size is not None):
            
            idx_valid = __split_by_class(df, col_target, valid_size, black_list = idx_train + idx_test)

            df_valid = df.iloc[idx_valid]
            
            df_valid.reset_index(drop=True, inplace=True)
            return (df_train[columns_train], df_train[col_target], 
                    df_test[columns_train], df_test[col_target],
                    df_valid[columns_train], df_valid[col_target])

def make_coords(coord_limit_x, coord_limit_y, target, len_set=10):
    '''
    Esta função gera -len_set- coordenadas com valores entre os limites coord_limit_x e coord_limit_y
    '''
    conjunto = []
    for i in range(len_set):
        x = random.uniform(coord_limit_x[0], coord_limit_x[1]) #randrange
        y = random.uniform(coord_limit_y[0], coord_limit_y[1]) #randrange
        conjunto.append((x,y,target))
        
    return conjunto


def normalize_col(df, col_name):
    '''
    >>> df_iris = normalize_col(df_iris, 'sepal length (cm)')
    '''
    max_ = df[col_name].max()
    min_ = df[col_name].min()
    
    df[col_name] = df[col_name].apply(lambda old_row: ((old_row - min_) / (max_ - min_)) )
    
    return df
    
def select_columns_p_value(features, target, columns, value=0.05):
    '''
    Adapted from: https://www.kaggle.com/bbloggsbott/feature-selection-correlation-and-p-value
    '''
    numVars = len(features[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(target, features).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > value:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    features = np.delete(features, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return columns