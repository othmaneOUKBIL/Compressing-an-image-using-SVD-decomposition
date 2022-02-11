# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:00:20 2022

@author: O-OUK
"""
import numpy as np
import statistics
from matplotlib import pyplot as plt
from PIL import Image

#transformer l'image en tableau de deux dimansion 
im = Image.open("C:/Users/O-OUK/OneDrive/Bureau/lena_gris1.png")
T = np.asarray(im)


# h,l=T.shape #hauteur, largeur de l'image
# cc=Image.fromarray(T)

#applicatin de la SVD sur la matrice de l'image
u, sigma, vh = np.linalg.svd(T)


#la fonction pour obtenir la matrice mére
def recombine(u, sigma, vt, ind):
    S = np.zeros((len(u), len(vt[0])))
    for i in range(ind):
        S[i][i] += sigma[i]
    # print(S)
    A = np.dot(u, S)
    A = np.dot(A, vt)
    return A

#appliquer la methode de svd pour pllusieur valeur de k 
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
    cc = recombine(u, sigma, vh, i)
    cc1 = Image.fromarray(cc)
    cc1.show()

plt.figure(1)
plt.title('les valeurs singuliers en fonction de k',color='red')
plt.xlabel('les valeurs de k',color='blue')
plt.ylabel('les valeurs singuliers',color='blue')
#plt.plot([i for i in range(395)],sigma,'k-',5,color='green')
#calculer la variance de sigma
var=statistics.variance(sigma)

#Déterminer la valeur de k permettant de capter 95% de la variance
k=0
index=0
sommesigma=0
val=0
for  i in range(len(sigma)):
    sommesigma+=sigma[i]

while index < len(sigma):
    k+=sigma[index]
    if (k**2)/(sommesigma**2)>0.95:
        val=index
        index=len(sigma)+1
    index+=1
print('la valeur de k permettant de capter 95% de la variance : ',val)

#Afficher  la compression via SVD pour cette valeur.
cc = recombine(u, sigma, vh, val)
cc1 = Image.fromarray(cc)
cc1.show()
