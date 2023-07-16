from models.Rejection import *
from src import metrics, utils

def _get_model(model_name, rejection_cost, type_model, threshold_decision, list_threshold):
    if model_name == 'classifier_bayes':
        return Rejection(rejection_cost, model_name, type_model, threshold_decision, list_threshold)
        
        
        
def pipeline(df,
             model_name,
             rejection_cost, 
             type_model='binary', 
             threshold_decision=0.5,
             list_threshold=[0.10, 0.15, 0.20, 0.25, 0.30],
             verbose_train=False,
             col_target='target',
             train_size=0.8,
             n_realizations=20):
    
    dict_all_realizations = {}
    
    list_acc = []
    
    list_reject_rate = []
    
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
        
        model = _get_model(model_name, rejection_cost, type_model, threshold_decision, list_threshold)
        
        model.fit(X_train, y_train, verbose_train)
        
        y_pred = model.predict(X_test)
        
        dict_all_realizations[f'realization_{i}']['model'] = model
        
        df_train = X_train.copy()
        df_train['target'] = y_train
        dict_all_realizations[f'realization_{i}']['train'] = df_train
        
        df_test = X_test.copy()
        df_test['target'] = y_test        
        dict_all_realizations[f'realization_{i}']['test'] = df_test
        
        # Desconsiderando a opções de rejeição para computo da acurácia
        X_test['label'] = y_test
        X_test['predict'] = y_pred
        
        qt_reject = len(X_test[X_test['predict'] == -9999])
        reject_rate = qt_reject / len(X_test)
        
        X_test = X_test[X_test['predict'] != -9999].copy()
        if (len(X_test) == 0): print('All of the test set were rejected.')
        y_pred, y_test = X_test['predict'], X_test['label']
        
        acc_realization = metrics.accuracy(y_test, y_pred)
        list_acc.append(acc_realization)
        list_reject_rate.append(reject_rate)
        mtx_confusion = metrics.confusion_matrix(y_test, y_pred)
        
        
        dict_all_realizations[f'realization_{i}']['acc'] = acc_realization
        dict_all_realizations[f'realization_{i}']['reject_rate'] = reject_rate
        dict_all_realizations[f'realization_{i}']['mtx_confusion'] = mtx_confusion
        
        print('acc: ', acc_realization)
        print('reject_rate: ', reject_rate)
        print('matrix confusion:\n', mtx_confusion)
        print('\n')
        
        
    accuracy = sum(list_acc) / n_realizations
    avg_reject_rate = sum(list_reject_rate) / n_realizations
    std_acc = metrics.std(list_acc)
    std_reject_rate = metrics.std(list_reject_rate)
    
    print('*' * 10, 'Final Result', '*' * 10)
    print('Acurracy: ', accuracy)
    print('Std Acurracy: ', std_acc)
    print('Avg reject rate: ', avg_reject_rate)
    print('Std Reject Rate: ', std_reject_rate)
     
    
    return dict_all_realizations, accuracy, std_acc, avg_reject_rate, std_reject_rate


def select_best_realization(dict_):
    
    best_realization = dict_[f'realization_1']
    for n in range(2, len(dict_) + 1):
        realization = dict_[f'realization_{n}']
        if realization['acc'] > best_realization['acc']:
            best_realization = realization
            
    return best_realization