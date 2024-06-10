import numpy as np
from koregraph.utils.pickles import load_pickle_object
from koregraph.config.params import GENERATED_PICKLE_DIRECTORY, MODEL_OUTPUT_DIRECTORY
from koregraph.api.preprocessing.posture_proc import generate_posture_array



# #Definir la custom_loss basee sur le calcul du delta
# def smooth_mse(y_true, y_pred):
#     y_true = y_true.reshape(-1, 17, 2)
#     y_pred = y_pred.reshape(-1, 17, 2)
#     #delta_mse = np.mean(np.abs(np.diff(y_pred, n=1, axis=0))**2)
#     delta_mse = np.sqrt(np.sum(np.diff(y_pred, n=1, axis=0)**2, axis=1))
#     print(delta_mse)
#     #mse = np.mean((y_true - y_pred)**2)
#     #res = mse + delta_mse
#     return delta_mse #,res

# #Pb1 : load data from pickles : pq 'EOFError: Ran out of input'?
# #Pb2 : pred : pq j ai des data audio?
# #Pb3 : quelle pred correspond a quel true ?
# #PB4 : passer cette custom_loss dans le modele. J'ai voulu le mettre dans le train_workflow, mais je ne vois pas où est la fonction de loss


# y_true = load_pickle_object('/Users/maudb/code/MaudBenichou/Projet/Koregraph/koregraph/generated/inputs/generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl')
# print(y_true)
# y_pred = load_pickle_object('/Users/maudb/code/MaudBenichou/Projet/Koregraph/koregraph/generated/predictions/model_sBM_cAll_d00_mBR2_ch02.pkl')
# print(y_pred)
# B = smooth_mse(y_true, y_pred)
# print(B)
#Fonctionnement du calcul de distance euclidienne entre la posture i et la posture i+1
y_pred = generate_posture_array('gBR_sBM_cAll_d04_mBR0_ch01.pkl') #charger un dataset shape(720, 17, 2)
A = np.diff(y_pred, n=1, axis=0) #calculer les distances  pour chaqun des 17 joints, sur x et sur y sur l'ensemble des 720 postures (719 distances en x, pareil en y) shape(719, 17, 2)
print(f'A={A}')
B = A**2 #passer les deltas x et deltas y au carré
print(f'B={B}')
C = np.zeros((B.shape[0], B.shape[1])) #C array vide à remplir dans la boucle qui suit
for b in range(B.shape[0]): #pour chacune des 719 deltas, on somme pour chaqun des 17 joints deltaX^2 + deltaY^2
    C[b] = np.sum(B[b], axis=1) #On obtient, pour chaque joint, la distance euclidienne au carré entre la posture i et la posture i+1
print(f'C={C}')
print(C.shape)
D = np.sqrt(C) #On prend la racine pour obtenir la distance au carré.
print(f'D={D}')
print(D.shape)
