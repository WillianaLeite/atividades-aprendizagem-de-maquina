import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_decision_boundary(df, list_features, model, h=.1, col_target='target', task='binary_classification'):
    '''
    Essa função funciona somente para duas features e com até 3 classes
    
    df: Dataframe pandas
    list_features: Lista com features que serão analisadas
    model: Modelo previamente treinado, que possua uma função predict
    h: Tamanho do passo no meshgrid
    col_target: Nome da coluna que representa variável resposta
    task=Pode assumir 'classifier' ou 'regressor'
    Ex: 

    >>> plot_decision_boundary(
        
        df = df_artificial_1, 
        list_features = ['x', 'y'],
        col_target='target',
        model = clf,
        h=.1    
    )
    '''
    def transform_hot(y):
        
        return y.index(1)
    
    if task == 'binary_classification':
        df = df[list_features + [col_target]]
        X = df.values[:, :2]
        y = df[col_target]
        list_light = ['orange', 'cornflowerblue', 'cyan']
        list_bold = ['darkorange', 'darkblue', 'c']
        
        qtd_classes = len(df[col_target].unique())
        
        cmap_light = ListedColormap(list_light[:qtd_classes])
        cmap_bold = list_bold[:qtd_classes]
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list_features))
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12, 10))
        plt.contourf(xx, yy, Z, cmap=cmap_light)
    
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df[col_target], palette=cmap_bold, alpha=1.0, edgecolor="black")
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"Superficie de Decisão")
        plt.xlabel(list_features[0])
        plt.ylabel(list_features[1])
    
        plt.show()
    
    elif task == 'multiclass_classification':
        
        df = df[list_features + [col_target]]
        X = df.values[:, :2]
        y = df[col_target]
        list_light = ['orange', 'cornflowerblue', 'cyan']
        list_bold = ['darkorange', 'darkblue', 'c']
        
        qtd_classes = len(df[col_target].unique())
        
        cmap_light = ListedColormap(list_light[:qtd_classes])
        cmap_bold = list_bold[:qtd_classes]
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list_features))
        ##Z = [ transform_hot(y) for y in Z]
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12, 10))
        plt.contourf(xx, yy, Z, cmap=cmap_light)
    
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df[col_target], palette=cmap_bold, alpha=1.0, edgecolor="black")
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"Superficie de Decisão")
        plt.xlabel(list_features[0])
        plt.ylabel(list_features[1])
    
        plt.show()
    
    elif task == 'regressor':
        
        if len(list_features) == 1:
            
            x_min, x_max, = df["".join(list_features)].min(), df["".join(list_features)].max()
            xx = np.arange(x_min, x_max, h)
            
            list_predict = model.predict(pd.DataFrame(xx.tolist(), columns=list_features))
            
            plt.figure(figsize=(12, 10))
                
            sns.scatterplot(x=xx, y=list_predict, alpha=1.0, edgecolor="black")
                    
            plt.xlim(xx.min(), xx.max())
            plt.ylim(min(list_predict), max(list_predict))
            plt.title(f"Superficie de Decisão")
            plt.xlabel('x')
            plt.ylabel('y')
                
            plt.show()
            
        elif len(list_features) == 2:
            
            df = df[list_features + [col_target]]
            X = df.values[:, :2]
            
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            
            x1, x2 = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            list_predict = model.predict(pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=list_features))
            
            threedee = plt.figure(figsize=(12, 10)).gca(projection='3d')
        
            threedee.scatter(x1, x2, list_predict)
            threedee.set_xlabel('x1')
            threedee.set_ylabel('x2')
            threedee.set_zlabel('target')
            plt.show()

    
    
def plot_error_epoch(lista):
    
    df = pd.DataFrame(data=lista, columns=['epoch', 'error'])
    
    df.plot.line( x='epoch', y='error', figsize=(10,6))
    
    plt.show()