import numpy as np
from koregraph.utils.pickles import load_pickle_object


#Definir la custom_loss basee sur le calcul du delta
def smooth_mse(y_true, y_pred):
    delta_mse = np.mean(np.abs(np.diff(y_pred, n=1, axis=0))**2)
    mse = np.mean((y_true - y_pred)**2)
    res = mse + delta_mse
    return res

#Pb1 : load data from pickles : pq 'EOFError: Ran out of input'?
#Pb2 : pred : pq j ai des data audio?
#Pb3 : quelle pred correspond a quel true ?
#PB4 : passer cette custom_loss dans le modele. J'ai voulu le mettre dans le train_workflow, mais je ne vois pas où est la fonction de loss


y_true = load_pickle_object('/Users/maudb/code/MaudBenichou/Projet/Koregraph/koregraph/generated/inputs/generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl')
print(y_true)
y_pred = load_pickle_object('/Users/maudb/code/MaudBenichou/Projet/Koregraph/koregraph/generated/predictions/model_sBM_cAll_d00_mBR2_ch02.pkl')
print(y_pred)
B = smooth_mse(y_true, y_pred)
print(B)
