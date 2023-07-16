import pandas as pd
import numpy as np
import json
from models.ClassifierBayes import *

class Rejection():
    
    def __init__(self, rejection_cost, model='classifier_bayes', type_model='binary', threshold_decision=0.5, list_threshold=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
        self.wr = rejection_cost
        self.best_t = 0
        self.threshold_decision = threshold_decision
        self.list_threshold = list_threshold
        if model == 'classifier_bayes':
            self.model = ClassifierBayes()
        if type_model != 'binary':
            raise ValueError('This implementation only works for binary problems.')

    def fit(self, X_train, y_train, verbose=False):
        if len(list(set(y_train))) > 2:
            raise ValueError('This implementation only works for binary problems.')
        else:
            self.model.fit(X_train, y_train)
            self.df_prob = self.model.predict_proba(X_train.copy())
            self.df_prob['label'] = y_train
            self.qt_classes = len(self.model.list_class)
            return self.__minimize_rejection(verbose=verbose)

    def __minimize_rejection(self, verbose=False):

        iterations = {}
        min_t = np.inf
        min_sum = np.inf
        for t in self.list_threshold:
            qt_reject = 0 
            erro = 0
            for idx, row in (self.df_prob.iterrows()):
                label = row['label']
                prob_class_0 = row[f'proba_0']
                if (prob_class_0 >= self.threshold_decision - t) and (prob_class_0 <= self.threshold_decision + t):
                    qt_reject += 1
                elif prob_class_0 < (self.threshold_decision - t): 
                    class_predict = 1
                    if class_predict != label: erro += 1
                elif prob_class_0 > (self.threshold_decision + t): 
                    class_predict = 0
                    if class_predict != label: erro += 1

            iterations[f't={t}'] = {
                'r(t)': qt_reject / len(self.df_prob),
                'e(t)': erro / (len(self.df_prob) - qt_reject)
            }
            iterations[f't={t}']['sum'] = iterations[f't={t}']['e(t)'] + (self.wr * iterations[f't={t}']['r(t)'])
            if iterations[f't={t}']['sum'] < min_sum:
                min_sum = iterations[f't={t}']['sum']
                min_t = t

        if min_t != np.inf: self.best_t = min_t
        if verbose: 
            print(json.dumps(iterations, indent=4))
            print(f'Best t: {self.best_t}')

    def predict(self, df):

        df_prob = self.model.predict_proba(df.copy())
        list_predict = []
        for idx, row in df_prob.iterrows(): 
            prob_first_class = row[f'proba_0']
            if (self.best_t != np.inf) and (self.best_t != 0):
                if (prob_first_class >= self.threshold_decision - self.best_t) and (prob_first_class <= self.threshold_decision + self.best_t):
                    class_predict = -9999 
                elif prob_first_class < (self.threshold_decision - self.best_t): 
                    class_predict = 1
                elif prob_first_class > (self.threshold_decision + self.best_t): 
                    class_predict = 0
                    
            else:
                if prob_first_class < self.threshold_decision: class_predict = 1
                else: class_predict = 0

            list_predict.append(class_predict)
        
        return list_predict