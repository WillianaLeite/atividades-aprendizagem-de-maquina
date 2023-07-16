from models.ClassifierBayesDiscriminant import *
from src import metrics, utils

def _get_model(model_name, discrimant, type_cov):
    if model_name == 'classifier_bayes_discriminant':
        return ClassifierBayesDiscriminant(discrimant, type_cov)
        
        
        
def pipeline(df,
             model_name,
             discrimant='quadratico', 
             type_cov='pool',
             col_target='target',
             train_size=0.8,
             n_realizations=20):
    
    dict_all_realizations = {}
    
    list_acc = []
    
    for i in range(1, n_realizations+1):
        
        print(f'Realization {i}')
        dict_all_realizations[f'realization_{i}'] = {}
        
        df = df.sample(frac=1).reset_index(drop=True).copy() #Shuffle
        
        if train_size is None:
            X_train, y_train = df.drop([col_target], axis=1).copy(), df[col_target]
            X_test, y_test = X_train.copy(), y_train
            
        else:
            X_train, y_train, X_test, y_test = utils.split_train_test(df.copy(), col_target=col_target, 
                                                                      train_size=train_size, stratify=True)
        
        model = _get_model(model_name, discrimant, type_cov)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc_realization = metrics.accuracy(y_test, y_pred)
        list_acc.append(acc_realization)
        mtx_confusion = metrics.confusion_matrix(y_test, y_pred)
        
        dict_all_realizations[f'realization_{i}']['model'] = model
        
        df_train = X_train.copy()
        df_train['target'] = y_train
        dict_all_realizations[f'realization_{i}']['train'] = df_train
        
        df_test = X_test.copy()
        df_test['target'] = y_test        
        dict_all_realizations[f'realization_{i}']['test'] = df_test
        
        dict_all_realizations[f'realization_{i}']['acc'] = acc_realization
        dict_all_realizations[f'realization_{i}']['mtx_confusion'] = mtx_confusion
        
        print('acc: ', acc_realization)
        print('matrix confusion:\n', mtx_confusion)
        print('\n\n')
        
        
    accuracy = sum(list_acc) / n_realizations
    std = metrics.std(list_acc)
    
    print('*' * 10, 'Final Result', '*' * 10)
    print('Acurracy: ', accuracy)
    print('Std: ', std) 
    
    return dict_all_realizations


def select_best_realization(dict_):
    
    best_realization = dict_[f'realization_1']
    for n in range(2, len(dict_) + 1):
        realization = dict_[f'realization_{n}']
        if realization['acc'] > best_realization['acc']:
            best_realization = realization
            
    return best_realization