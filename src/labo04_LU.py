#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
import sys
sys.path.append(".") # poner path
from labo00_auxiliares import *

def calculaLU(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for iter in range(n):
        pivot = Ac[iter][iter]
        if pivot == 0:
            return None, None, 0
        for fila in range(iter+1,n):
            L_i_inv = Ac[fila][iter] / pivot                                            ; cant_op += 1
            Ac[fila][iter] = L_i_inv 
            Ac[fila][iter+1:n] = -L_i_inv * Ac[iter][iter+1:n] + Ac[fila][iter+1:n]     ; cant_op += (n-(iter+1))*2 # para mi era *3 no *2 pero fallan tests 
    
    L = triangularInferior(Ac) + np.identity(n)                       ; # estas no cuenan cant_op += n**2, si no fallan tests

    U = Ac - triangularInferior(Ac) 



                
            
    
    return L, U, cant_op


A = np.array([
    [2,1,2,3],
    [4,3,3,4],
    [-2,2,-4,-12],
    [4,1,8,-3],
]   )


def res_tri(L, b, inferior = True) :
    if not inferior:
        L = traspuestaPorOtraDiagonal(L)
        b = b[::-1]
    n = len(b)
    x = np.zeros(n)
    for i, row in enumerate(L):
        x[i] = (b[i] - prodint(row[:i], x[:i]) )/row[i]
    
    
    return x if inferior else x[::-1]
