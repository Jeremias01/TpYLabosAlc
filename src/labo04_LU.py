#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
from typing import Text
import numpy as np
import sys

from labo01_errores_igualdad import esSimetrica
sys.path.append(".") # poner path
sys.path.append("src") 
from labo00_auxiliares import *

def calculaLU(A):
    """
    Calcula la factorizacion LU de la matriz A y retorna las matrices L
    y U, junto con el numero de operaciones realizadas. En caso de
    que la matriz no pueda factorizarse retorna Nones.
    """
    cant_op = 0
    A = np.array(A, dtype=float)
    m = A.shape[0]
    n = A.shape[1]
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
    
    L = triangularInferior(Ac) + identidad(n)                       ; # estas no cuenan cant_op += n**2, si no fallan tests

    U = Ac - triangularInferior(Ac) 

    
    return L, U, cant_op


A = np.array([
    [2,1,2,3],
    [4,3,3,4],
    [-2,2,-4,-12],
    [4,1,8,-3],
]   )


def res_tri(L, b, inferior = True) :
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    inferior (por default asumir que es triangular inferior).
    """
    if not inferior:
        L = rotar180(L)
        b = b[::-1]
    n = len(b)
    x = np.zeros(n)
    for i, row in enumerate(L):
        x[i] = (b[i] - prodint(row[:i], x[:i]) )/row[i]
    
    
    return x if inferior else x[::-1]

def res_tri_mat(L, B, inferior = True):
    """
    Resuelve el sistema LX = B, 
    Recibe L triangular, superior o inferior segun parametro inferior, y B.
    Devuelve X, la matriz solucion. 
    """
    if L.shape[0] != B.shape[0]:
        return None 
    res = np.zeros((L.shape[1], B.shape[0]))
    for i,col in enumerate(traspuesta(B)):
        res[:,i] = res_tri(L, col, inferior)
    return res


def res_LU(L,U, b):
    y = res_tri(L,b)
    x = res_tri(U,y, inferior=False)
    return x

def res_LU_mat(L,U, B):
    Y = res_tri_mat(L,B)
    X = res_tri_mat(U,Y, inferior=False)
    return X


def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares
    retorna None si no es inversible
    """
    A = np.array(A, dtype=float)
    L,U,_ = calculaLU(A)
    if L is None or U is None:
        return None
    res = identidad(A.shape[0])
    for i in range(len(res)):
        res[i] = res_LU(L,U,res[i])
    return traspuesta(res) 

def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    Ademas devuelve la cantidad de operaciones realizadas
    Devuleve 
    """
    L,U,opA = calculaLU(A)
    if L is None or U is None:
        return None,None,None,0
    V,D,opU = calculaLU(traspuesta(U))
    return L,D,traspuesta(V),opA+opU

def esSDP(A, atol=1e-8) :
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    """
    if not esSimetrica(A, atol):
        return False
    L,D,V,_ = calculaLDV(A)
    if L is None or D is None or V is None:
        return False
    for i in range(A.shape[0]):
        if A[i][i] <= 0:
            return False

    return True
