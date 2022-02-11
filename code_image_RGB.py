# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 00:29:42 2022

@author: O-OUK
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#transformer l'image en tableau de deux dimansion 
im = Image.open("lien d'image ")
T = np.asarray(im)
R=T[:,:,0]
G=T[:,:,1]
B=T[:,:,2]

# h,l=T.shape #hauteur, largeur de l'image
# cc=Image.fromarray(T)

#applicatin de la SVD sur la matrice de l'image
def ssvd(T):
    u, sigma, vh = np.linalg.svd(T)
    return u,sigma,vh


#la fonction pour obtenir la matrice mére
def recombine(M, ind):
    u,sigma,vt=np.linalg.svd(M)
    S=np.diag(sigma)
    A = np.dot(u[:,:ind], S[:ind,:ind])
    A = np.dot(A, vt[:ind,:])
    return A

#appliquer la methode de svd pour pllusieur valeur de k 
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
    R1 = recombine(R, i)
    G1=recombine(G, i)
    B1=recombine(B,i)
    M=np.zeros((435,395,3))
    
    M[:,:,0]=R1
    M[:,:,1]=G1
    M[:,:,2]=B1
    M=M.astype(np.uint8)
    arr = Image.fromarray(M)
    arr.show()
