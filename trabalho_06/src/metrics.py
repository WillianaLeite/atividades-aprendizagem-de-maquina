from math import sqrt
import numpy as np

def confusion_matrix(y_true, y_pred):
    
    list_class = list(set(y_true))
    matrix_confusion = np.zeros((len(list_class), len(list_class)))
    
    for desejado, predito in zip(y_true,y_pred):
        matrix_confusion[desejado][predito] +=1                
            
    return matrix_confusion


def accuracy(y_true, y_pred, from_mtx_confusion=False): # Taxa de acerto
    
    if from_mtx_confusion:
        
        mtx_confusion = confusion_matrix(y_true, y_pred)
        total = np.sum(mtx_confusion)
        total_acerto = 0
        for class_ in list(set(y_true)):
            total_acerto += mtx_confusion[class_][class_] # Pegando a diagonal
        
        return total_acerto / total
      
    else:
    
        qtd_acertos = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred: 
                qtd_acertos += 1

        return qtd_acertos / len(y_true)
    
def std(lista):
    
    mean = sum(lista) / len(lista)

    variation = sum([(num-mean) ** 2 for num in lista]) / (len(lista) - 1) 

    return sqrt(variation)